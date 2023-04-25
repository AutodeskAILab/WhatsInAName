import hashlib
import json
import os
import re
from argparse import ArgumentParser
from ast import literal_eval
from distutils.util import strtobool
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from abcpartnames.transforms.lower_and_replace_transform import LowerAndReplace_Transform


def remove_random_item_from_list(items):
    k = torch.randint(len(items), (1,))
    target = items[k]
    chosen_items = items[:k] + items[k + 1:]
    return chosen_items, target


def load_idx_file(idx_path, split):
    with open(idx_path, 'r') as f:
        subset = json.load(f)
    if split in ['train', 'validation', 'test']:
        keys = subset[split]
    elif split == 'all':
        keys = subset['train'] + subset['validation'] + subset['test']
    else:
        raise ValueError('split must be \'train\', \'validation\', \'test\' or \'all\'')
    return keys


class ABCPairs(Dataset):
    def __init__(self,
                 input_dir='data/abc',
                 split='train',
                 transform=None,
                 remove_=False,
                 lower=False, **kwargs) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.remove_ = remove_
        self.lower = lower

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f'split must be \'train\', \'validation\' or \'test\' - received: \'{split}\'')

        self.df = pd.read_csv(self.input_dir / f'{split}_pairs.csv', header=None)
        self.labels = torch.from_numpy(self.df[0].values)
        self.a = self.df[1].values
        self.b = self.df[2].values

    def __getitem__(self, index) -> T_co:
        a = self.a[index]
        b = self.b[index]

        if self.remove_:
            a = a.replace('_', ' ')
            b = b.replace('_', ' ')

        if self.lower:
            a = a.lower()
            b = b.lower()

        if self.transform:
            a = self.transform(a)
            b = self.transform(b)

        return a, b, self.labels[index]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group('ABCPairsData')
        parser.add_argument('--remove_', type=strtobool, default=False,
                            help='replace underscores with spaces')
        parser.add_argument('--lower', type=strtobool, default=True,
                            help='convert strings to lower case')
        return parent_parser


class TechNetPairs(ABCPairs):
    def __init__(self, input_dir='data/abc', split='train', transform=None, lower=None, **kwargs) -> None:
        super(Dataset).__init__()
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.lower = False

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f'split must be \'train\', \'validation\' or \'test\' - received: \'{split}\'')

        self.df = pd.read_csv(self.input_dir / f'{split}_pairs_technet.csv', header=None)
        self.labels = torch.from_numpy(self.df[0].values)
        self.a = self.df[3].apply(literal_eval).values
        self.b = self.df[4].apply(literal_eval).values


class ABCPairsDataModule(pl.LightningDataModule):
    def __init__(self, split, batch_size=32, num_workers=0, transform=None, **kwargs):
        super().__init__()
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.dset = ABCPairs(split=self.split, transform=self.transform, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        n_test = int(0.1 * len(self.dset))
        n_val = int(0.1 * len(self.dset))
        n_train = len(self.dset) - n_val - n_test
        self.train, self.val, self.test = random_split(self.dset, [n_train, n_val, n_test])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=os.cpu_count() if self.num_workers == -1 else self.num_workers,
                          multiprocessing_context='spawn' if self.num_workers > 0 else None)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=os.cpu_count() if self.num_workers == -1 else self.num_workers,
                          multiprocessing_context='spawn' if self.num_workers > 0 else None)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group('ABCPairsData')
        parser.add_argument('--input_dir', type=str, default='data/abc',
                            help='input directory containing abc data')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='dataloader batch size')
        parser.add_argument('--num_workers', type=int, default=-1,
                            help='number of dataloader workers - use -1 for num cpus')
        parent_parser = ABCPairs.add_argparse_args(parent_parser)
        return parent_parser


class ABCBaseDataModule(pl.LightningDataModule):

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        val_size = int(len(self.full_train_dataset) * 0.15)
        train_size = len(self.full_train_dataset) - val_size
        self.train, self.val = torch.utils.data.random_split(
            self.full_train_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate,
                          pin_memory=True,
                          num_workers=os.cpu_count() if self.num_workers == -1 else self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # Note:  Do we need to update this code to do the 
        #        final evaluation?
        return DataLoader(dataset=self.full_validation_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate,
                          num_workers=0)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate,
                          num_workers=os.cpu_count() if self.num_workers == -1 else self.num_workers)


class ABCNamesWithPartsDataModule(ABCBaseDataModule):
    def __init__(self, batch_size=32, num_workers=0, transform=None, pred_names=False, no_lower=False, replace_=False,
                 model=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.lower = not no_lower
        self.pred_names = pred_names
        self.replace_ = replace_
        self.collate = ABCNamesWithParts.collate
        self.model = model

        if self.pred_names:
            self.idx_path = 'data/abc/train_val_test_partnames.json'
        else:
            self.idx_path = 'data/abc/train_val_test_two_or_more_partnames.json'

        # Note:  Do we need to update this code to do the
        #        final evaluation?
        self.full_train_dataset = self.construct_dataset('train')
        self.full_validation_dataset = self.construct_dataset('validation')

    def construct_dataset(self, split):
        if self.model == 'technet':
            return TechnetABCNamesWithParts(
                idx_path=self.idx_path,
                pred_names=self.pred_names,
                split=split,
                lower=self.lower,
                replace_=self.replace_
            )
        else:
            return ABCNamesWithParts(
                json_path='data/abc/abc_text_data_003.json',
                idx_path=self.idx_path,
                pred_names=self.pred_names,
                split=split,
                lower=self.lower,
                replace_=self.replace_
            )

    def create_caching_encoder(
            self,
            model_name,
            model_options,
            model_supplier
    ):
        dataset_options = {
            "dataset_type": str(type(self.full_train_dataset)),
            "pred_names": self.pred_names,
            "no_lower": not self.lower,
            "replace_": self.replace_
        }
        all_data = dataset = torch.utils.data.ConcatDataset(
            [self.full_train_dataset, self.full_validation_dataset]
        )
        return CachedNamesAndParts(
            all_data,
            model_name=model_name,
            model_options=model_options,
            dataset_options=dataset_options,
            model_supplier=model_supplier
        )

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parent_parser = ABCNamesWithParts.add_argparse_args(parent_parser)
        return parent_parser


class ABCUniqueStrings(Dataset):
    def __init__(self, input_dir='data/abc', split='train', transform=None, with_raw=False) -> None:
        super().__init__()
        self.input_dir = Path(input_dir)
        self.transform = transform or (lambda _: _)
        self.with_raw = with_raw

        if split not in ['train', 'val', 'test']:
            raise ValueError(f'split must be \'train\', \'val\' or \'test\' - received: \'{split}\'')

        with open(self.input_dir / f'{split}.json', 'r') as f:
            data = json.load(f)
        unique = set()
        for key, parts in data.items():
            unique.update(parts)
        self.unique = sorted(unique)

    def __getitem__(self, index) -> T_co:
        if self.with_raw:
            return self.transform(self.unique[index]), self.unique[index]
        return self.transform(self.unique[index])

    def __len__(self):
        return len(self.unique)


class ABCNamesWithParts(Dataset):
    def __init__(self, json_path: str = 'data/abc/abc_text_data_003.json',
                 idx_path: str = None,
                 split='all', pred_names=True, replace_=False, lower=True, target_transform=None) -> None:
        super().__init__()
        self.pred_names = pred_names
        self.lower_replace_transform = LowerAndReplace_Transform(replace_, lower)
        self.target_transform = target_transform
        # load input data
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if idx_path:
            self.keys = load_idx_file(idx_path, split)
        else:
            self.keys = list(self.data.keys())

    def __getitem__(self, idx):
        key = self.keys[idx]
        document = self.data[key]
        name = document['document_name']
        parts = document['body_names']

        if self.pred_names:
            # We are trying to predict the name.
            # We give the document name as the
            # target and leave the name blank
            target = document['document_name']
            name = None
        else:
            # We are removing one part name from
            # parts and trying to predict this as
            # the target
            parts, target = remove_random_item_from_list(parts)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if name is not None:
            name = self.lower_replace_transform(name)
        target = self.lower_replace_transform(target)
        parts = self.lower_replace_transform(parts)

        return name, parts, target

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def collate(batch):
        names, parts, targets = zip(*batch)
        return names, parts, targets

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group('ABCNamesWithParts')
        parser.add_argument('--pred_names', type=strtobool, default=False,
                            help='predict assembly names using all parts as input')
        parser.add_argument('--no_lower', action='store_true', default=False,
                            help='do not convert strings to lower case')
        parser.add_argument('--replace_', action='store_true', default=False,
                            help='Replace _ characters with spaces')
        return parent_parser


class TechnetABCNamesWithParts(ABCNamesWithParts):
    def __init__(self, json_path: str = 'data/abc/abc_text_data_technet_003.json', idx_path: str = None, split='all',
                 pred_names=True, replace_=False, lower=True, target_transform=None) -> None:
        super().__init__(json_path, idx_path, split, pred_names, replace_, lower, target_transform)

    def __getitem__(self, idx):
        key = self.keys[idx]
        document = self.data[key]
        name = document['document_name']
        parts = document['body_names']

        if self.pred_names:
            # We are trying to predict the name.
            # We give the document name as the
            # target and leave the name blank
            target = document['document_name']
            name = None
        else:
            # We are removing one part name from
            # parts and trying to predict this as
            # the target
            parts, target = remove_random_item_from_list(parts)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if name is not None:
            name = [self.lower_replace_transform(n) for n in name]
        target = [self.lower_replace_transform(t) for t in target]
        parts = [self.lower_replace_transform(p) for p in parts]

        return name, parts, target


class CachedNamesAndParts:
    def __init__(
            self,
            dataset,
            model_name: str,
            model_options: dict,
            dataset_options: dict,
            model_supplier):
        self.dataset = dataset
        cache_name = self.get_cache_name(model_name, model_options, dataset_options)
        self.cache_path = f'cache/{cache_name}'
        if os.path.exists(self.cache_path + '.npy'):
            print(f'loading cache from {self.cache_path}')
            self.embs = torch.from_numpy(joblib.load(self.cache_path + '.npy', mmap_mode='r').copy())
            self.lookup = joblib.load(self.cache_path + '.lookup')
        else:
            print('data cache not found')
            self.fill_cache(model_supplier)

    def get_cache_name(self, model_name, model_options, dataset_options):
        model_options_str = json.dumps(model_options)
        dataset_options_str = json.dumps(dataset_options)
        string_to_hash = model_name + model_options_str + dataset_options_str
        return hashlib.sha224(string_to_hash.encode("UTF-8")).hexdigest()

    def __getitem__(self, item) -> torch.Tensor:
        idx = self.lookup[item]
        return self.embs[idx].float()

    def __call__(self, item) -> torch.Tensor:
        return self[item]

    def __len__(self):
        return len(self.lookup)

    def fill_cache(self, model_supplier):
        print('instantiating model...')
        model = model_supplier()
        embs = []
        lookup = {}
        idx = 0
        for string_data in tqdm(self.dataset, desc='filling cache'):
            for s in string_data:
                if isinstance(s, list) or isinstance(s, tuple):
                    for part in s:
                        embs.append(model(part))
                        lookup[part] = idx
                        idx += 1
                elif isinstance(s, str):
                    embs.append(model(s))
                    lookup[s] = idx
                    idx += 1
                elif s is None:
                    # It's OK if the name is null
                    pass
                else:
                    assert False, f"Unknown data {type(s)}"

        self.embs = torch.stack(embs)
        self.lookup = lookup
        print(f'saving cache to {self.cache_path}...')
        joblib.dump(self.embs.detach().cpu().numpy(), self.cache_path + '.npy')
        joblib.dump(self.lookup, self.cache_path + '.lookup')


class ABCNamesPartsDescriptionsFeatures(Dataset):
    def __init__(self,
                 json_path: str = 'data/abc/abc_text_data_003.json',
                 idx_path: str = None,
                 split: str = 'train') -> None:
        super().__init__()
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if idx_path:
            with open(idx_path, 'r') as f:
                subset = json.load(f)
            if split in ['train', 'validation', 'test']:
                self.keys = subset[split]
            elif split == 'all':
                self.keys = subset['train'] + subset['val'] + subset['test']
            else:
                raise ValueError('split must be \'train\', \'val\', \'test\' or \'all\'')
        else:
            self.keys = list(self.data.keys())

    def __getitem__(self, index) -> T_co:
        return self.data[self.keys[index]]

    def __len__(self):
        return len(self.keys)
