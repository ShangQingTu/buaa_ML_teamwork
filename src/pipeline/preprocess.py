import torch
from transformers import *
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class DataProcessor(object):
    def __init__(self, tokenizer, max_sequence_length, truncation_strategy='longest_first'):
        """
        :arg
         - tokenizer: 分词器
         - max_sequence_length: 拼接后句子的最长长度
         - truncation_strategy: 如果句子超过max_sequence_length,截断的策略
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.truncation_strategy = truncation_strategy
        self.label2id = {'neutral': 0, 'positive': 1, 'negative': 2}

    def _convert_to_transformer_inputs(self, text):
        """
        Converts tokenized input to ids, masks and segments for transformer (including bert)
        :arg
         - text: 文本

        :return
         - input_ids: 记录句子里每个词对应在词表里的 id
         - input_masks: 列表中， 1的部分代表句子单词，而后面0的部分代表paddig，只是用于保持输入整齐，没有实际意义。
           相当于告诉BertModel不要利用后面0的部分
        - segments: 列表用来指定哪个是第一个句子，哪个是第二个句子，0的部分代表句子一, 1的部分代表句子二
        """

        inputs = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_sequence_length,
                                            truncation_strategy=self.truncation_strategy,
                                            # truncation=True
                                            )

        input_ids = inputs["input_ids"]
        input_segments = inputs["token_type_ids"]
        input_masks = [1] * len(input_ids)
        padding_length = self.max_sequence_length - len(input_ids)
        padding_id = self.tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return input_ids, input_masks, input_segments

    def get_input(self, df):
        """
        :param
         - df: 数据集集的dataFrame

        :return:
            3个处理好的tensor,形状都是[数据总数,max_sequence_length],它们的含义请看_convert_to_transformer_inputs
         - tokens_tensor: (tensor) [数据总数,max_sequence_length]
         - input_masks_tensors: (tensor) [数据总数,max_sequence_length]
         - segments: 列表用来指定哪个是第一个句子，哪个是第二个句子，0的部分代表句子一, 1的部分代表句子二
        """
        token_ids, masks, input_segments = [], [], []
        # 每一条数据
        for i in tqdm(range(len(df))):
            if i == 0:
                # 不要读到表头
                continue
            text = df.iloc[i]['text']
            input_ids, input_masks, input_segment = self._convert_to_transformer_inputs(text)
            token_ids.append(input_ids)
            masks.append(input_masks)
            input_segments.append(input_segment)

        tokens_tensor = torch.tensor(token_ids)
        input_masks_tensors = torch.tensor(masks)
        input_segments_tensors = torch.tensor(input_segments)

        return [tokens_tensor, input_masks_tensors, input_segments_tensors]

    def get_output(self, df_train):
        """
        :param df_train: 训练集的dataFrame
        :return: (tensor) [num_vocab] 数据的标注,只有0和1,1代表这个reply回答了query
        """
        labels = df_train['labels']
        id_labels = []
        for label in tqdm(labels):
            if label == 'labels':
                continue
            id_label = self.label2id[label]
            id_labels.append(id_label)
        # return torch.tensor(np.array(id_labels)).unsqueeze(1)
        return torch.tensor(np.array(id_labels))
