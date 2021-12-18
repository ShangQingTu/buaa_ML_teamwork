import itertools
import warnings
from transformers.data.processors.utils import InputExample, InputFeatures
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer
import json


class TokenizerWrapper:
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 create_token_type_ids: Optional[str] = False,
                 **kwargs):
        self.max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        if truncate_method == 'tail':
            self.truncate_fct = self.truncate_from_tail
        elif truncate_method == 'head':
            self.truncate_fct = self.truncate_from_head
        else:
            raise NotImplementedError

        self.create_token_type_ids = create_token_type_ids

        self.template_mask_token = '<mask>'
        self.template_eos_token = '<eos>'
        self.template_bos_token = '<bos>'
        self.template_sep_token = '<sep>'
        self.template_cls_token = '<cls>'
        self.template_pad_token = '<pad>'

        from transformers import logging
        verbosity_before = logging.get_verbosity()
        logging.set_verbosity(logging.CRITICAL)  # TODO solve this in a more elegant way
        self.mask_token_map = {
            self.template_mask_token: self.tokenizer.mask_token if hasattr(self.tokenizer, 'mask_token') else ''}
        self.eos_token_map = {
            self.template_eos_token: self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') else ''}
        self.bos_token_map = {
            self.template_bos_token: self.tokenizer.bos_token if hasattr(self.tokenizer, 'bos_token') else ''}
        self.sep_token_map = {
            self.template_sep_token: self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else ''}
        self.cls_token_map = {
            self.template_cls_token: self.tokenizer.cls_token if hasattr(self.tokenizer, 'cls_token') else ''}
        self.pad_token_map = {
            self.template_pad_token: self.tokenizer.pad_token if hasattr(self.tokenizer, 'pad_token') else ''}
        logging.set_verbosity(verbosity_before)

        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def truncate_rate(self, ):
        r"""Using this function, one can easily identify how many sentence has be truncated, thus help the user to choose a better thresthold for chunking.
        """
        if self.total_passed_sentences == 0:
            return None
        else:
            return self.num_truncated_sentences / self.total_passed_sentences

    @property
    def special_tokens_maps(self, ) -> Dict:
        r"""This need to be specified in specific language model
        """
        if not hasattr(self, "_special_tokens_map"):
            _special_tokens_map = {}
            for attrname in self.__dict__.keys():
                if attrname.endswith('_token_map'):
                    _special_tokens_map.update(getattr(self, attrname))
        return _special_tokens_map

    def tokenize_with_mask(self,
                           wrapped_example: List[Dict],
                           ) -> InputFeatures:
        raise NotImplementedError

    def tokenize_without_mask(self,
                              wrapped_example: List[Dict],
                              ) -> InputFeatures:
        raise NotImplementedError

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                           num_tokens_to_truncate: int = 0) -> Dict:
        r"""truncate the inputs from the rear
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts[::-1]):
                if len(part) == 0:  # to prevent some part are empty after tokenization
                    continue
                if shortenable_ids[-1 - i][0] == 0:  # ==0 means the part is not shortenable
                    continue
                parts[-1 - i] = part[:-to_trunc] if to_trunc < len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def truncate_from_head(input_dict: Dict,
                           num_tokens_to_truncate: int = 0) -> Dict:
        r"""truncate the inputs from the head
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts):
                if shortenable_ids[i][0] == 0:  # ==0 means the part is not shortenable
                    continue
                parts[i] = part[:-to_trunc] if to_trunc < len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    @staticmethod
    def padding(input_dict: Dict,
                max_len: int, pad_id_for_inputs: int = 0, pad_id_for_others: int = 0) -> None:
        for key, value in input_dict.items():
            if (len(input_dict[key]) > max_len):
                raise ValueError(f'''
                    Truncated seq length of '{key}' still greater than max length '{max_len}.'
                    One possible reason is that no enough shortenable parts in template. Try add {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:
                input_dict[key].extend([pad_id_for_inputs] * (max_len - len(value)))
            else:
                input_dict[key].extend([pad_id_for_others] * (max_len - len(value)))
        return input_dict

    def add_special_tokens(self, encoder_inputs):
        # add special tokens
        for key in encoder_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
                        encoder_inputs[key])
            else:
                special_tokens_mask = np.array(self.tokenizer.get_special_tokens_mask(encoder_inputs[key]))
                with_special_tokens = np.array(self.tokenizer.build_inputs_with_special_tokens(encoder_inputs[key]))
                if key in ["soft_token_ids"]:  # TODO maybe more than this
                    encoder_inputs[key] = ((1 - special_tokens_mask) * with_special_tokens).tolist()  # use 0 as special
                else:
                    encoder_inputs[key] = ((
                                                   1 - special_tokens_mask) * with_special_tokens - special_tokens_mask * 100).tolist()  # use -100 as special
        return encoder_inputs

    def truncate(self, encoder_inputs):
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences += 1
        if num_tokens_to_truncate > 0:
            self.num_truncated_sentences += 1
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                                               num_tokens_to_truncate=num_tokens_to_truncate)
        return encoder_inputs


class MLMTokenizerWrapper(TokenizerWrapper):
    add_input_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_ids(self):
        return self.tokenizer.mask_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        '''
        '''

        wrapped_example, others = wrapped_example

        # for some dataset like SuperGLUE.COPA, the answer requires prediction an span of
        # the input. Or in generation tasks, we need to generate a piece of target_text.
        # In these case, it tokenized to the encoded_tgt_text for furture use.
        encoded_tgt_text = []
        if 'tgt_text' in others:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]
            for t in tgt_text:
                encoded_tgt_text.append(self.tokenizer.encode(t, add_special_tokens=False))

        mask_id = 0  # the i-th the mask token in the template.

        encoder_inputs = defaultdict(list)
        for piece in wrapped_example:
            if piece['loss_ids'] == 1:
                if teacher_forcing:  # fill the mask with the tgt task
                    raise RuntimeError("Masked Language Model can't perform teacher forcing training!")
                else:
                    encode_text = [self.mask_token_ids]
                mask_id += 1

            if piece['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[piece['text']]
                if to_replace is not None:
                    piece['text'] = to_replace
                else:
                    raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

            if 'soft_token_ids' in piece and piece['soft_token_ids'] != 0:
                encode_text = [0]  # can be replace by any token, since these token will use their own embeddings
            else:
                encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)

            encoding_length = len(encode_text)
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text']:
                    encoder_inputs[key].append([piece[key]] * encoding_length)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        # create special input ids
        encoder_inputs['attention_mask'] = [1] * len(encoder_inputs['input_ids'])
        if self.create_token_type_ids:
            encoder_inputs['token_type_ids'] = [0] * len(encoder_inputs['input_ids'])
        # padding
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length,
                                      pad_id_for_inputs=self.tokenizer.pad_token_id)

        if len(encoded_tgt_text) > 0:
            encoder_inputs = {**encoder_inputs, "encoded_tgt_text": encoded_tgt_text}  # convert defaultdict to dict
        else:
            encoder_inputs = {**encoder_inputs}
        return encoder_inputs


class InputFeatures(dict):
    """
    The class for input to the PLM and Prompts. To make users explicitly know the available keys,
    we define a dict with a set of predefined possible keys. The default value to any key is None.
    When use it as a dict, all the keys whose values are None are invisible.
    This class support most of the dict's operation (See Examples). It can also be consumed by
    pytorch's default_collate in DataLoader.
    Also a :py:meth:`to_tensor()` method is build to convert the values into torch.Tensor for torch's input.
    Examples:
    ..  code-block:: python
        in_feat = InputFeatures(**{'input_ids':[1,4,5], 'soft_token_ids': [3,4,5]})  # init from dict
        print(in_feat.keys())       # ['input_ids, 'soft_token_ids']
        in_feat['label'] = 3        # can assign value like normal dict
        print(in_feat.keys())       # ['input_ids','label', 'soft_token_ids'] (Note that it's also ordered)
        print(in_feat['label'])     # 3
        in_feat['alice'] = 0        # KeyError: Key alice not in predefined set of keys
        in_feat.values()            # [[1,4,5], 3, [3,4,5]]  (Note that it's also ordered)
        [in_feat[key] for key in in_feat]   # [[1,4,5], 3, [3,4,5]]
        new_dict= {**in_feat, 'new_key':2}  # new_dict is {'input_ids': [1, 4, 5], 'label': 3, 'soft_token_ids': [3, 4, 5], 'new_key': 2}
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
                       'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
                       'past_key_values', 'loss_ids']
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
                'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
                'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len']
    non_tensorable_keys = []

    def __init__(self,
                 input_ids: Optional[Union[List, torch.Tensor]] = None,
                 inputs_embeds: Optional[torch.Tensor] = None,
                 attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
                 token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
                 label: Optional[Union[int, torch.Tensor]] = None,
                 decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
                 decoder_inputs_embeds: Optional[torch.Tensor] = None,
                 soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
                 past_key_values: Optional[torch.Tensor] = None,  # for prefix_tuning
                 loss_ids: Optional[Union[List, torch.Tensor]] = None,
                 guid: Optional[str] = None,
                 tgt_text: Optional[str] = None,
                 use_cache: Optional[bool] = None,
                 encoded_tgt_text: Optional[str] = None,
                 input_ids_len: Optional[int] = None,
                 **kwargs):
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len

        for k in kwargs.keys():
            setattr(self, k, kwargs[k])

    def to_tensor(self, device: str = 'cuda'):
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self, keep_none=False) -> Dict[str, Any]:
        """get the dict of mapping from keys to values of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`Dict[str, Any]`: dict of mapping from keys to values of the InputFeatures
        """
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] = value
            elif value is None and keep_none:
                data[key] = None
        return data

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    @staticmethod
    def collate_fct(batch: List):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''

        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    print(f"key{key}\n d {[batch[i][key] for i in range(len(batch))]} ")

        return InputFeatures(**return_dict)


class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always neccessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid=None,
                 text_a="",
                 text_b="",
                 label=None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str, List[str]]] = None
                 ):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]
