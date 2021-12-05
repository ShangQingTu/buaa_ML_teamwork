# coding=UTF-8
import torch
import torch.utils.data as Data
import argparse
from transformers import AutoModel, BertTokenizer
from tqdm import tqdm
from torch import nn, optim
import random
import numpy as np
import shutil
import os
import pandas as pd
import logging
from sklearn.metrics import f1_score

import src.models.albert
from src.pipeline.preprocess import DataProcessor
from src.models.albert import AlbertClassifierModel
from src.pipeline.utils import setup_logger, MetricLogger, strip_prefix_if_present


def train(inputs, outputs, args, logger):
    """
     :param:
     - inputs: (list) 作为输入的tensor, 它是由get_input处理得的
     - outputs: (tensor) 作为标注的tensor, 它是由get_output处理得的
     - args: 一堆训练前规定好的参数
     - logger: 训练日志,可以把训练过程记录在./ckpt/log.txt
     :return: 训练结束
     """
    # 创建数据集
    torch_dataset = Data.TensorDataset(inputs[0], inputs[1], inputs[2], outputs)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
    logger.info('[1] Building model')
    # 查看运行训练脚本时,所用的设备,如果有cuda,就用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构造 model
    model = AlbertClassifierModel(num_topics=args.num_topics,
                                  out_channels=args.out_channels,
                                  max_input_len=args.max_input_len,
                                  kernel_size=args.kernel_size,
                                  dropout=args.dropout).to(device)
    model_kwargs = {k: getattr(args, k) for k in
                    {'num_topics', 'out_channels', 'max_input_len', 'kernel_size', 'dropout'}
                    }
    logger.info(model)
    # 优化器
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    meters = MetricLogger(delimiter="  ")
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # scheduler,在schedule_step的时候,把学习率乘0.1,目前只在第一个step做了这个下降
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.schedule_step], gamma=0.1)
    logger.info('[2] Start training......')
    for epoch_num in range(args.max_epoch):
        # example_num:一个epoch需要训练多少个batch
        example_num = outputs.shape[0] // args.batch_size
        for batch_iter, (input_ids, attention_mask, input_segments, label) in enumerate(loader):
            progress = epoch_num + batch_iter / example_num
            optimizer.zero_grad()
            # 正向传播
            pred = model(input_ids.to(device),
                         segment_ids=input_segments.to(device),
                         attention_mask=attention_mask.to(device))
            # 处理 label
            if label.shape[0] != args.batch_size:
                logger.info('last dummy batch')
                break
            label = label.to(device)
            # print(pred)
            # print(pred.shape)
            # print(label)
            # print(label.shape)
            loss = criterion(pred, label)

            # 反向传播
            loss.backward()
            optimizer.step()
            meters.update(loss=loss)
            # 每过0.01个epoch记录一次loss
            if (batch_iter + 1) % (example_num // 100) == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "progress: {prog:.2f}",
                            "{meters}",
                        ]
                    ).format(
                        prog=progress,
                        meters=str(meters),
                    )
                )
        # 验证这个epoch的效果
        score = validate(inputs, outputs, model, device, args)
        logger.info("val result score:")
        logger.info(score)
        save = {
            'kwargs': model_kwargs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step()
        # 每个epoch保留一个ckpt
        torch.save(save,
                   os.path.join(args.save_dir, 'model_epoch%d_val%.3f.pt' % (epoch_num, score)))


def validate(inputs, outputs, model, device, args):
    """
    :param
    - inputs: (list) 作为输入的tensor, 它是由get_input处理得的
    - outputs: (tensor) 作为标注的tensor, 它是由get_output处理得的
    - args: 一堆训练前规定好的参数
    :return:
    - f1: 最好的f1分数
    """
    with torch.no_grad():
        torch_dataset = Data.TensorDataset(inputs[0], inputs[1], inputs[2], outputs)
        # radom choose
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
        pred_probs = []
        labels = []
        for batch_iter, (input_ids, attention_mask, segments_tensor, label) in enumerate(loader):
            # 测2000条来验证这个epoch的效果
            if batch_iter * args.batch_size > 2000:
                break
            pred_prob = model(input_ids.to(device), segments_tensor.to(device), attention_mask.to(device))
            pred_probs.append(pred_prob)
            labels.append(label.int())
        # df_preds = pd.DataFrame()
        print("pred_probs")
        print(len(pred_probs))
        merged_pred_probs = torch.cat(pred_probs).cpu().numpy()
        print(merged_pred_probs.shape)
        df_preds = pd.DataFrame(np.argmax(merged_pred_probs, axis=1))
        df_labels = pd.DataFrame(torch.cat(labels).view(-1).cpu().numpy())
        print("df_preds")
        print(df_preds.shape)
        print(df_labels.shape)
        f1 = f1_score(df_labels, df_preds, average="micro")
    return f1


def work(args):
    """
    :param args: 一堆训练前规定好的参数
    :return: 训练结束
    """
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    df_train = pd.read_csv(args.df_train_path, header=None)
    df_train.columns = ['index', 'text', 'labels', 'id']
    processor = DataProcessor(tokenizer, args.max_input_len)
    inputs = processor.get_input(df_train)
    outputs = processor.get_output(df_train)
    print(inputs[0].shape)
    print(inputs[1].shape)
    print(outputs.shape)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("Classify", args.save_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(inputs, outputs, args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-model', type=str, default=src.models.albert.pretrained)
    # model parameters, see them in `model.py`
    parser.add_argument('--num-topics', type=int, default=3)
    parser.add_argument('--max-input-len', type=int, default=100)
    parser.add_argument('--out-channels', type=int, default=2)
    parser.add_argument('--kernel-size', type=int, nargs='+', default=[2, 3, 4])

    # training parameters
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optim', default='adam', choices=['adam', 'adamw', 'sgd'])
    args = parser.parse_args()
    args.df_train_path = os.path.join(args.data_dir, "Train.csv")
    args.df_test_path = os.path.join(args.data_dir, "Test.csv")
    work(args)
