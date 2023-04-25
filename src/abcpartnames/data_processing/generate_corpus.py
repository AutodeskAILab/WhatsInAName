from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from tqdm import tqdm

from abcpartnames.datasets.ABCTextDataset import ABCNamesPartsDescriptionsFeatures


def write_corpus(lower, dset, target_path, idx):
    with open(target_path, 'w') as f:
        for i in tqdm(idx):
            document = dset[i]

            name = document['document_name']
            description = document['document_description']
            bodies = document['body_names']
            features = document['feature_names']

            name_str = f'An assembly with name "{name}"'
            description_str = f' and description "{description}",' if description else ''
            parts_str = f' contains the following parts: {", ".join(bodies + features)}.'

            line = name_str + description_str + parts_str + '\n'

            if lower:
                line = line.lower()
            f.write(line)


def write_corpus_parts_only(lower, dset, target_path, idx, remove_duplicates):
    with open(target_path, 'w') as f:
        for i in tqdm(idx):
            document = dset[i]

            bodies = document['body_names']

            if remove_duplicates:
                bodies = set(bodies)

            line = ", ".join(bodies) + '.\n'

            if lower:
                line = line.lower()
            f.write(line)


def main(args):
    np.random.seed(args.seed)

    dset = ABCNamesPartsDescriptionsFeatures(json_path=args.json_path,
                                             idx_path=args.idx_path,
                                             split=args.split)

    idx = np.arange(len(dset))
    np.random.shuffle(idx)

    print('writing data')
    if args.parts_only:
        write_corpus_parts_only(lower=args.lower,
                                dset=dset,
                                target_path=args.out,
                                idx=idx,
                                remove_duplicates=args.remove_duplicates)
    else:
        write_corpus(lower=args.lower,
                     dset=dset,
                     target_path=args.out,
                     idx=idx)

    print('completed')


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_path', type=str, default='data/abc/abc_text_data_003.json',
                        help='train.json file containing all train data')
    parser.add_argument('--idx_path', type=str, default='data/abc/train_val_test_split.json',
                        help='json file containing idx of splits')
    parser.add_argument('--split', type=str, default='train',
                        help='split to create corpus from (\'train\', \'validation\' or \'test\'')
    parser.add_argument('--out', type=str, default='data/abc/train_corpus_complete.txt',
                        help='directory to output generated files')
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed for split')
    parser.add_argument('--lower', action='store_true', default=False,
                        help='convert all text to lower case')
    parser.add_argument('--parts_only', action='store_true', default=False,
                        help='only write the parts as a comma separated list')
    parser.add_argument('--remove_duplicates', action='store_true', default=False,
                        help='remove duplicate parts when \'--parts_only\' is set')
    main(parser.parse_args())
