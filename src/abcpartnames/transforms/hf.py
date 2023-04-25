import atexit
import os.path
import pickle
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

CACHE = {}


class SentenceEncoding(ABC):
    @abstractmethod
    def __call__(self, out):
        pass


class MeanPooling(SentenceEncoding):

    def __init__(self, layer=-2) -> None:
        super().__init__()
        self.layer = layer

    def __call__(self, out):
        # hidden states is layer x batch x token number x hidden units
        embedding_layer = out.hidden_states[self.layer]
        return embedding_layer.mean(dim=1).squeeze()


class ClsToken(SentenceEncoding):

    def __init__(self, layer=-2) -> None:
        super().__init__()
        self.layer = layer

    def __call__(self, out):
        # hidden states is layer x batch x token number x hidden units
        embedding_layer = out.hidden_states[self.layer]
        return embedding_layer[0][0]


class ToPreTrainedMaskedLMEmbedding:
    def __init__(self, model, device, with_cache=True, sentence_method: SentenceEncoding = MeanPooling(-2)) -> None:
        global CACHE
        self.model_name = model.replace('.', '#').replace('/', '#')
        self.with_cache = with_cache
        self.sentence_method = sentence_method or MeanPooling(layer=-2)

        if with_cache:
            os.makedirs('cache/ToPreTrainedMaskedLMEmbedding', exist_ok=True)
            self.cache_path = f'cache/ToPreTrainedMaskedLMEmbedding/{self.model_name}.embeddings'
            if os.path.exists(self.cache_path):
                print(f'loading embeddings cache from file: {self.cache_path}')
                with open(self.cache_path, 'rb') as f:
                    CACHE = pickle.load(f)
            else:
                CACHE = {}
                atexit.register(self.save_cache)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
        self.model.eval()

    def __call__(self, text) -> torch.Tensor:
        with torch.no_grad():
            if self.with_cache:
                if text not in CACHE.keys():
                    CACHE[text] = self.get_embedding(text)
                return CACHE[text]
            else:
                return self.get_embedding(text)

    def get_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
        out = self.model(**tokens, return_dict=True, output_hidden_states=True, output_attentions=True)
        return self.sentence_method(out)

    def save_cache(self):
        print(f'saving embeddings cache: {self.cache_path}')
        with open(self.cache_path, 'wb') as f:
            pickle.dump(CACHE, f)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('ABCPairsData')
        parser.add_argument('--sentence_method', type=str, default='mean',
                            help='method for representing sentence: \'mean\' = mean pool, \'cls\' = use CLS token')
        parser.add_argument('--sentence_layer', type=int, default=-2,
                            help='layer of hidden dims to use for sentence embedding')
        return parent_parser


class MaskedLMEncoder:
    def __init__(self, model, device) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
        self.model.eval()

    def __call__(self, text) -> torch.Tensor:
        return self.get_embedding(text)

    def get_embedding(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, tuple):
            texts = list(texts)
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
        out = self.model(**tokens, return_dict=True, output_hidden_states=True, output_attentions=True)
        return self.mean_pool(out, mask=tokens.data['attention_mask'])

    @staticmethod
    def mean_pool(out, mask):
        # hidden states is layer x batch x token number x hidden units
        embedding_layer = out.hidden_states[-2]  # type: torch.Tensor
        if mask is not None:
            masked_embeddings = embedding_layer * mask[:, :, None]
            mean = masked_embeddings.sum(dim=1) / mask.sum(-1)[:, None]
        else:
            mean = embedding_layer.mean(dim=1)
        return mean
