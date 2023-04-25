import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import joblib
from nltk import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from abcpartnames.datasets.ABCTextDataset import ABCNamesWithParts


def main(args):
    cache_dir = Path(args.cache)
    os.makedirs(cache_dir, exist_ok=True)

    dset = ABCNamesWithParts(json_path=args.input,
                             idx_path=args.idx_path)

    dataloader = DataLoader(dataset=dset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=ABCNamesWithParts.collate,
                            num_workers=0)

    tokenizer = WordPunctTokenizer()

    vectorizers = {
        'Count': CountVectorizer(stop_words='english',
                                 tokenizer=tokenizer.tokenize),
        'Tf-IDF': TfidfVectorizer(stop_words='english',
                                  tokenizer=tokenizer.tokenize)
    }

    docs = []
    for i, (names, parts) in tqdm(enumerate(dataloader), total=len(dset) // args.batch_size, desc='prepare data'):
        docs += [
            f'{doc_name}: {", ".join(doc_parts)}'
            for doc_name, doc_parts in zip(names, parts)
        ]

    for name, vectorizer in vectorizers.items():
        print(f'fitting {name}...')
        vectorizer.fit(docs)
        path = cache_dir / f'{name}.joblib'
        print(f'saving to {path}')
        joblib.dump(vectorizer, path)

    print('complete')


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default='data/abc/abc_text_data_003.json',
                        help='json file containing all ABC text data')
    parser.add_argument('--idx_path', type=str, default='data/abc/train_val_test_two_or_more_partnames.json',
                        help='idx file containing splits')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for data loader')
    parser.add_argument('--cache', type=str, default='cache/vectorizers',
                        help='directory to save vectorizers')
    main(parser.parse_args())
