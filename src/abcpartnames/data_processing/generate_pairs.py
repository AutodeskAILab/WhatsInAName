import argparse
import json
import os
import random
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk import wordpunct_tokenize
from tqdm import tqdm


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    nltk.download('popular')

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    X = {}
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    with open(args.idx_path, 'r') as f:
        idx = json.load(f)

    for split in ['train', 'validation', 'test']:
        X[split] = {document_id: data[document_id]['body_names'] for document_id in idx[split]}

    true_pairs = {}
    for split in ['validation', 'test']:
        print(f'selecting true pairs: {split}')

        true_pairs[split] = []
        for key, parts in tqdm(X[split].items()):
            shuffled_parts = random.sample(parts, k=len(parts))
            first, *rest = shuffled_parts

            first_tokens = set(wordpunct_tokenize(first.lower()))

            for part_string in rest:
                rest_tokens = set(wordpunct_tokenize(part_string.lower()))
                if first_tokens.isdisjoint(rest_tokens):
                    true_pairs[split].append((first, part_string))
                    break
        print(f'created {len(true_pairs[split])}/{len(X[split])}')

    print('creating co-occurrence table')
    # create hashtable mapping each part to a set of it's co-occurring parts
    all = X['train']
    all.update(X['validation'])
    all.update(X['test'])
    table = {}
    for key in tqdm(all.keys()):
        parts = all[key]
        for part in parts:
            if part not in table.keys():
                table[part] = set()
            others = set(parts)
            others.remove(part)
            table[part].update(others)

    false_pairs = {}
    for split in ['validation', 'test']:
        print(f'creating false pairs: {split}')
        false_pairs[split] = []
        a, b = zip(*true_pairs[split])
        b_ = random.sample(b, k=len(b))
        for first, last in zip(a, b_):
            if last not in table[first]:
                false_pairs[split].append((first, last))
        print(f'created {len(false_pairs[split])}/{len(true_pairs[split])}')

        if len(false_pairs[split]) < len(true_pairs[split]):
            print(f'dropping true pairs from {split} to balance sizes')
            true_pairs[split] = true_pairs[split][:len(false_pairs[split])]

    print(f'saving pairs')
    for split in ['validation', 'test']:
        a_true, b_true = zip(*true_pairs[split])
        a_false, b_false = zip(*false_pairs[split])
        df = pd.DataFrame({
            'label': [1] * len(true_pairs[split]) + [0] * len(false_pairs[split]),
            'a': a_true + a_false,
            'b': b_true + b_false
        })
        print(f'{split} (true/false/total): {len(true_pairs[split])}/{len(false_pairs[split])}/{len(df)}')
        df.to_csv(out_dir / f'{split}_pairs.csv', index=False, header=False)

    print('completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_path', type=str, default='data/abc/abc_text_data_003.json',
                        help='json file containing all documents')
    parser.add_argument('--idx_path', type=str, default='data/abc/train_val_test_two_or_more_partnames.json',
                        help='json file containing train/val/test splits for documents with two or more parts')
    parser.add_argument('--out_dir', type=str, default='data/abc',
                        help='output directory for generated files')
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed')
    main(parser.parse_args())
