from transformers import AutoModel
import torch
import torch.nn as nn
from torch.nn.functional import softmax

pretrained = 'bert-base-cased'


class AlbertClassifierModel(nn.Module):
    """
    注: 需要先学习一下ALBERT 和 TextCNN会比较好理解
    AlbertClassifierModel:
    把 Albert作为词向量,把句子映射为词向量的列表,
    然后用TextCNN提取特征,再加上Albert本身的特征向量CLS,
    连接之后,就是一个[1,embedding_size + out_channels * len(kernel_size)]维度的向量
    输入到 第一个全连接层fc_layer1, 并dropout一次,向量维度不变
    再输入到 第二个全连接层fc_layer1,输出为1个值(二分类的概率)
    """

    def __init__(self, num_topics, out_channels, max_input_len, kernel_size, dropout):
        """
        :arg:
        num_topics = 3 分类的类数
        out_channels = 2 卷积核 的数量
        kernel_size = [2,3,4] 卷积核 的 大小
        max_text_len = 100 句子最长长度
        dropout = 0.3 对输入的一些维度进行随机遗忘的比例,为了防止过拟合
        :var
        embedding_size = 768 词向量的维度
        """
        super(AlbertClassifierModel, self).__init__()
        self.ntopic = num_topics
        self.albert_model = AutoModel.from_pretrained(pretrained)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768,
                                    out_channels=out_channels,
                                    kernel_size=h),
                          # Sequential可以把卷积层、 激活函数层和池化层都包含进去
                          nn.LeakyReLU(),  # 激活函数层
                          nn.MaxPool1d(kernel_size=max_input_len - h + 1))
            # 池化层，这里kernel_size的值应该不用进行这样多余的计算，我们的目的
            # 是将feature maps转为一行一列的向量，那么只要保证这里的kernel_size大于等于feature maps的行数就可以了
            for h in kernel_size  # 不同卷积层的kernel_size不一样，注意：这里的kernel_size和池化层的kernel_size是不同的
        ])  # 创建多个卷积层，包含了 图中的convolution、activation function和 maxPooling
        dim_h = 768 + out_channels * len(kernel_size)
        self.fc_layer1 = nn.Linear(dim_h, dim_h)
        self.fc_layer2 = nn.Linear(dim_h, self.ntopic)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids, attention_mask, use_cls=True):
        """
        :param
           - input_ids: 记录句子里每个词对应在词表里的 id
           - segments: 列表用来指定哪个是第一个句子，哪个是第二个句子，0的部分代表句子一, 1的部分代表句子二
           - input_masks: 列表中， 1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
           相当于告诉BertModel不要利用后面0的部分
           - use_cls: 是否使用albert的cls特征向量
        :return:
           - y: 对 输入做3分类,预测出的概率
        """
        if (use_cls):
            all_hidden = self.albert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=segment_ids)[0]

            cls_hidden = all_hidden[:, 0, :]
            embed_x = all_hidden.permute(0, 2, 1)  # 将矩阵转置
            out = [conv(embed_x) for conv in self.convs]  # 计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
            out = torch.cat(out, dim=1).squeeze(2)  # 对6 个univariate vectors进行拼接
            y = self.fc_layer1(torch.cat((cls_hidden, out), 1))
            y = self.dropout(y)
            y = self.fc_layer2(y)
        else:
            all_hidden = self.albert_model(input_ids=input_ids,
                                           token_type_ids=segment_ids, attention_mask=attention_mask)[0]
            embed_x = all_hidden.permute(0, 2, 1)  # 将矩阵转置
            mean_hidden = torch.mean(all_hidden, dim=1, keepdim=False)
            out = [conv(embed_x) for conv in self.convs]  # 计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
            out = torch.cat(out, dim=1).squeeze(2)  # 对6 个univariate vectors进行拼接
            y = self.fc_layer1(torch.cat((mean_hidden, out), 1))
            y = self.dropout(y)
            y = self.fc_layer2(y)
        return y
