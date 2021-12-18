import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from src.models.bilstm import BiLSTMSentiment
import numpy as np
import pandas as pd
from torchtext.legacy.data import BucketIterator, Dataset, Example, Field, TabularDataset

# torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
root = '/home/tsq/buaa_ML_teamwork'
os.environ["CUDA_VISIBLE_DEVICES"] = '6'


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch ' + str(epoch + 1)):
        sent, label = batch.text, batch.labels
        if model.use_gpu:
            sent = sent.to('cuda')
            label = label.to('cuda')
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        optimizer.zero_grad()
        # 正向传播
        pred = model(sent)
        pred_label = pred.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        # model.zero_grad()
        # print("label")
        # print(label)
        # print("pred", pred)
        loss = loss_function(pred, label)
        # print("label")
        # print(label)
        # print("pred", pred)
        # print("loss")
        # print(loss.shape)
        # print(loss)
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label = batch.text, batch.labels
        if model.use_gpu:
            sent = sent.to('cuda')
            label = label.to('cuda')
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    if 'Test' in name:
        with open("{}/ckpt/bilstm_test.txt".format(root), 'w') as fout:
            for _pred in pred_res:
                fout.write(str(int(_pred)))
                fout.write("\n")
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc * 100))
    return acc


def get_dataset(csv_data, id_field, text_field, label_field, test=False):
    fields = [('id', id_field), ('text', text_field), ('labels', label_field)]
    examples = []

    if test:
        for _id, text in tqdm(zip(csv_data['Unnamed: 0'], csv_data['text'])):
            examples.append(Example.fromlist([int(_id), text, 'neutral'], fields))
    else:
        for _id, text, label in tqdm(zip(csv_data['Unnamed: 0'], csv_data['text'], csv_data['labels'])):
            examples.append(Example.fromlist([int(_id), text, label], fields))
    return examples, fields


def load_sst(text_field, label_field, id_field, batch_size):
    # train, dev, test = TabularDataset.splits(path='{}/data/'.format(root), train='Train.csv',
    #                                          validation='Valid.csv', test='Test.csv', format='csv',
    #                                          fields=[('text', text_field), ('labels', label_field)])
    train_data = pd.read_csv('{}/data/Train.csv'.format(root))
    valid_data = pd.read_csv('{}/data/Valid.csv'.format(root))
    test_data = pd.read_csv('{}/data/Test.csv'.format(root))
    # 得到构建Dataset所需的examples和fields
    train_examples, train_fields = get_dataset(train_data, id_field, text_field, label_field)
    valid_examples, valid_fields = get_dataset(valid_data, id_field, text_field, label_field)
    test_examples, test_fields = get_dataset(test_data, id_field, text_field, label_field, True)

    # 构建Dataset数据集
    train = Dataset(train_examples, train_fields)
    dev = Dataset(valid_examples, valid_fields)
    test = Dataset(test_examples, test_fields)
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    id_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = BucketIterator.splits((train, dev, test),
                                                            # batch_sizes=(batch_size, len(dev), len(test)),
                                                            batch_sizes=(batch_size, batch_size, batch_size),
                                                            sort_key=lambda x: x.id,
                                                            # repeat=False, device=-1)
                                                            repeat=False)
    ## for GPU run
    #     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


EPOCHS = 4
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 32
timestamp = str(int(time.time()))
best_dev_acc = 0.0

text_field = Field(tokenize='basic_english', fix_length=128, lower=True)
label_field = Field(sequential=False)
id_field = Field(sequential=False)
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, id_field, BATCH_SIZE)

model = BiLSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),
                        label_size=3, use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if USE_GPU:
    model = model.cuda()

print('Load word embeddings...')
# # glove
# text_field.vocab.load_vectors('glove.6B.100d')

# word2vector
word_to_idx = text_field.vocab.stoi
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0
word2vec = load_bin_vec('{}/data/GoogleNews-vectors-negative300.bin'.format(root), word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word] - 1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)

model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.NLLLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "data", "results", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc * 100))
    dev_acc = evaluate(model, dev_iter, loss_function, 'Dev')
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm ' + out_dir + '/best_model' + '.pth')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
        # evaluate on test with the best dev performance model
        test_acc = evaluate(best_model, test_iter, loss_function, 'Test')
test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test')
