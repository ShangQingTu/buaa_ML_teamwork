import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import *
from transformers.utils.dummy_pt_objects import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
from collections import namedtuple
import inspect
from src.models.mlm import InputExample, InputFeatures


class ManualTemplate(nn.Module):
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self, text, tokenizer,
                 placeholder_mapping: dict = {'<text_a>': 'text_a', '<text_b>': 'text_b'}):
        super().__init__()
        self.text = text
        self.tokenizer = tokenizer
        self.placeholder_mapping = placeholder_mapping
        self.mixed_token_start = "{"
        self.mixed_token_end = "}"
        self.on_text_set()

    def on_text_set(self):
        """
        when template text was set

        1. parse text
        """
        self.text = self.parse_text(self.text)

    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i - 1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                mixed_token_cnt = 1  # { {} {} } nested support
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0: break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(
                        f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)

        return parsed

    def process_batch(self, batch):
        return batch

    def post_processing_outputs(self, outputs):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        return outputs

    def incorporate_text_example(self,
                                 example: InputExample
                                 ):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x: x)(
                    getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x: x)(example.meta[d['meta']])
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

    def wrap_one_example(self,
                         example: InputExample) -> List[Dict]:
        r'''Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value.
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as
        whether it's shortenable, whether it's the masked position, whether it's soft token, etc.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.

        Args:
            example (:obj:`InputExample`): An :py:class:`~openprompt.data_utils.data_utils.InputExample` object, which should have attributes that are able to be filled in the template.

        Returns:
            :obj:`List[Dict]`: A list of dict of the same length as self.text. e.g. ``[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]``
        '''

        if self.text is None:
            raise ValueError("template text has not been initialized")
        if isinstance(example, InputExample):
            text = self.incorporate_text_example(example)

            not_empty_keys = example.keys()
            for placeholder_token in self.placeholder_mapping:
                not_empty_keys.remove(
                    self.placeholder_mapping[placeholder_token])  # placeholder has been processed, remove
            not_empty_keys.remove('meta')  # meta has been processed

            keys, values = ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, "get_default_" + inputflag_name):
                    v = getattr(self, "get_default_" + inputflag_name)()
                    setattr(self, inputflag_name, v)  # cache
                else:
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialize.
                    Try using template.{} = [...] to initialize
                    or create an method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))

                if len(v) != len(text):
                    raise ValueError("Template: len({})={} doesn't match len(text)={}." \
                                     .format(inputflag_name, len(v), len(text)))
                values.append(v)
            wrapped_parts_to_tokenize = []
            for piece in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]
        else:
            raise TypeError("InputExample")

    def get_default_loss_ids(self) -> List[int]:
        '''Get the loss indices for the template using mask.
        e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask"}.'``,
        output is ``[0, 0, 0, 0, 1, 0]``.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]:

            - 1 for a masked tokens.
            - 0 for a sequence tokens.
        '''
        return [1 if 'mask' in d else 0 for d in self.text]

    def get_default_shortenable_ids(self) -> List[int]:
        """Every template needs shortenable_ids, denoting which part of the template can be trucate to fit
        the language model's ``max_seq_length``. Default: the input text is shortenable, while the template text and other
        special tokens are not shortenable.
        e.g. when self.text is ``'{"placeholder": "text_a"} {"placeholder": "text_b", "shortenable": False} {"meta": "word"} is {"mask"}.'``,
        output is ``[1, 0, 0, 0, 0, 0, 0]``.

        Returns:
            :obj:`List[int]`: A list of integers in the range ``[0, 1]``:
            - 1 for the input tokens.
            - 0 for the template sequence tokens.
        """
        idx = []
        for d in self.text:
            if 'shortenable' in d:
                idx.append(1 if d['shortenable'] else 0)
            else:
                idx.append(1 if 'placeholder' in d else 0)
        return idx


class ManualVerbalizer(nn.Module):
    def __init__(self, classes, label_words, tokenizer,
                 multi_token_handler: Optional[bool] = "first",
                 post_log_softmax: Optional[bool] = True):
        super().__init__()
        self.classes = classes
        self.label_words = label_words
        self.tokenizer = tokenizer
        self.post_log_softmax = post_log_softmax
        self.num_classes = len(classes)
        self.multi_token_handler = multi_token_handler
        self.generate_parameters()

    def generate_parameters(self):
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label]
                          + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                          for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label]
                     + [[0] * max_len] * (max_num_label_words - len(ids_per_label))
                     for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def gather_outputs(self, outputs):
        r""" retrieve useful output for the verbalizer from the whole model ouput
        By default, it will only retrieve the logits
        Args:
            outputs (:obj:`ModelOutput`) The output from the pretrained language model.
        Return:
            :obj:`torch.Tensor` The gathered output, should be of shape (``batch_size``,
            ``seq_len``, ``any``)
        """
        return outputs.logits

    def process_outputs(self,
                        outputs: torch.Tensor,
                        batch,
                        **kwargs):
        r"""By default, the verbalizer will process the logits of the PLM's
        output.
        Args:
            logits (:obj:`torch.Tensor`): The current logits generated by pre-trained language models.
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of the data.
        """

        return self.process_logits(outputs, batch=batch, **kwargs)

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        (1) Project the logits into logits of label words
        if self.post_log_softmax is True:
            (2) Normalize over all label words
            (3) Calibrate (optional)
        (4) Aggregate (for multiple label words)
        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """

        # project
        label_words_logits = self.project(logits,
                                          **kwargs)
        # Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)

        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)
            # convert to logits
            label_words_logits = torch.log(label_words_probs + 1e-15)

        # aggreate
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.
        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.
        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def handle_multi_token(self, label_words_logits, mask):
        r"""
        Support multiple methods to handle the multi tokens produced by the tokenizer.
        We suggest using 'first' or 'max' if the some parts of the tokenization is not meaningful.
        Can broadcast to 3-d tensor.

        Args:
            label_words_logits (:obj:`torch.Tensor`):

        Returns:
            :obj:`torch.Tensor`
        """
        if self.multi_token_handler == "first":
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            label_words_logits = label_words_logits - 1000 * (1 - mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == "mean":
            label_words_logits = (label_words_logits * mask.unsqueeze(0)).sum(dim=-1) / (
                    mask.unsqueeze(0).sum(dim=-1) + 1e-15)
        else:
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits


class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``
    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''

    def __init__(self,
                 plm: PreTrainedModel,
                 template: ManualTemplate,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch) -> torch.Tensor:
        r"""
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        if isinstance(batch, InputFeatures):
            batch = batch.to_dict()
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        # default batch size is 1
        for k, tensor in input_batch.items():
            input_batch[k] = tensor.unsqueeze(0)
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs


class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.
    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the lables to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''

    def __init__(self,
                 plm: PreTrainedModel,
                 template: ManualTemplate,
                 verbalizer: ManualVerbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False
                 ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self, ):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        batch):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).
        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        if isinstance(batch, InputFeatures):
            batch = batch.to_dict()
        # print("outputs.shape", outputs.shape)
        idx = torch.where(batch['loss_ids'] > 0)
        # print("idx", idx)
        outputs = outputs[:, torch.where(batch['loss_ids'] > 0), :]
        # outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch) -> torch.Tensor:
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the lable words (obtained by the current verbalizer).
        """
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer


class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`str`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`bool`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """

    def __init__(self,
                 dataset,
                 template: ManualTemplate,
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class,
                 verbalizer: Optional[ManualVerbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                 ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing
        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
        prepare_kwargs = {
            "max_seq_length": max_seq_length,
            "truncate_method": truncate_method,
            "decoder_max_length": decoder_max_length,
            "predict_eos_token": predict_eos_token,
            "tokenizer": tokenizer,
            **kwargs,
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # processs
        self.wrap()
        self.tokenize()

        sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=InputFeatures.collate_fct,
            drop_last=drop_last,
        )

    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer,
                                                           'wrap_one_example'):  # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset), desc='tokenizing'):
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(
                **self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing),
                **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self, ):
        return self.dataloader.__iter__()


def signature(f):
    r"""Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.

    Args:
        f (:obj:`function`) : the function to get the input arguments.

    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s
    """
    sig = inspect.signature(f)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    keywords = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    keywords = keywords[0] if keywords else None
    defaults = [
                   p.default for p in sig.parameters.values()
                   if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                      and p.default is not p.empty
               ] or None
    argspec = namedtuple('Signature', ['args', 'defaults',
                                       'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)
