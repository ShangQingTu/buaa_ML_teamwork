# coding=UTF-8
import torch
from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
import argparse
from collections import namedtuple
import random
import numpy as np
import shutil
import os
import pandas as pd
import logging
from sklearn.metrics import f1_score
import src.models.albert
from src.models.mlm import MLMTokenizerWrapper, InputExample
from src.models.prompt import ManualVerbalizer, ManualTemplate, PromptForClassification, PromptDataLoader
from src.pipeline.utils import setup_logger, MetricLogger, strip_prefix_if_present

classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "neutral",
    "positive"
]

dataset = [  # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid=0,
        text_a="Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid=1,
        text_a="The film was just fine.",
    ),
    InputExample(
        guid=2,
        text_a="The film was badly made.",
    ),
]
label_words = {
    "negative": ["bad"],
    "neutral": ["normal"],
    "positive": ["good", "wonderful", "great"],
}

labels = ["positive", "neutral", "negative"]

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model', 'wrapper'))

_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    })}


def load_plm(model_name, model_path, specials_to_add=None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`model_config`: The wrapper class of this plm.
    """
    model_class = _MODEL_CLASSES[model_name]
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name:  # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    return model, tokenizer, model_config, wrapper


def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.
    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.
    Returns:
        The resized model, The tokenizer with the added special tokens.
    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer


def work(args):
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} It was {"mask"}',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "negative": ["bad"],
            "positive": ["good", "wonderful", "great"],
        },
        tokenizer=tokenizer,
    )
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )
    data_loader = PromptDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )
    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim=-1)
            print(classes[preds])
    # predictions would be 2, 1, 0 for classes 'positive', 'neutral' ,'negative'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--save-dir', type=str, default='./ckpt')
    parser.add_argument('--pretrained-model', type=str, default=src.models.albert.pretrained)
    args = parser.parse_args()
    args.df_train_path = os.path.join(args.data_dir, "Train.csv")
    args.df_test_path = os.path.join(args.data_dir, "Test.csv")
    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    logger = setup_logger("Prompt", args.save_dir)
    work(args)
