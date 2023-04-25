import re
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import CountVectorizer


class ToVectorized:
    def __init__(self, vectorizer_path) -> None:
        super().__init__()
        self.vectorizer = joblib.load(vectorizer_path)  # type: CountVectorizer

    def __call__(self, x):
        if isinstance(x, str):
            x = [x]
        csr = self.vectorizer.transform(x)

        coo = csr.tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        values = torch.LongTensor(coo.data)

        return torch.sparse.LongTensor(indices, values, torch.Size(coo.shape))

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('ToVectorized')
        parser.add_argument('--vectorizer_path', type=str, required=True,
                            help='path to saved vectorizer')
        return parent_parser


class ToFastText:
    def __init__(self, gensim_path) -> None:
        super().__init__()
        assert gensim_path is not None, "Must define --gensim_path if using fasttext encoder"
        self.model = FastText.load(gensim_path)

    def __call__(self, x):
        word_embs = []
        for token in re.split(r'\s+', x):
            word_embs.append(self.model.wv[token])

        part_emb = np.mean(word_embs, axis=0)

        return torch.from_numpy(part_emb).float()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('ToVectorized')
        parser.add_argument('--gensim_path', type=str,
                            help='path to saved FastText model')
        return parent_parser


class ToTechNet:
    def __init__(self, root_dir='data/technet') -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        v1 = pd.read_csv(self.root_dir / 'vocab_github_1.tsv', sep='\t', header=None)
        v2 = pd.read_csv(self.root_dir / 'vocab_github_2.tsv', sep='\t', header=None)
        self.lookup = {}
        for key, file, line in pd.concat([v1, v2]).values:
            self.lookup[key] = (file, line)

    def __call__(self, clean_parts: List[str]):
        embs = []
        for part in clean_parts:
            try:
                file, line = self.lookup[part]
                arr = np.loadtxt(self.root_dir / f'word_embeddings_{file}.txt', skiprows=line, max_rows=1, dtype=object)
                word = arr[0]
                emb = arr[1:].astype(np.float)
                embs.append(torch.from_numpy(emb).float())
                assert word == part
            except KeyError:
                embs.append(torch.zeros(600, dtype=torch.float))
        if len(embs) == 0:
            return torch.zeros(600, dtype=torch.float)
        elif len(embs) == 1:
            return embs[0]
        else:
            return torch.mean(torch.stack(embs), dim=0)
