from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl

from abcpartnames.datasets.ABCTextDataset import ABCPairsDataModule, ABCPairs
from abcpartnames.models.mlp import MLP, SparseMLP
from abcpartnames.scripts.evaluate_model import evaluate
from abcpartnames.transforms.vectorizers import ToVectorized


def main(parser):
    parser = MLP.add_model_specific_args(parser)
    parser = ABCPairsDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ToVectorized.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(seed=args.seed + args.trial)

    transform = ToVectorized(args.vectorizer_path)
    args.embedding_size = len(transform.vectorizer.vocabulary_)

    evaluate(model=SparseMLP(args),
             datamodule=ABCPairsDataModule.from_argparse_args(args,
                                                              split='validation',
                                                              dset=ABCPairs,
                                                              transform=transform),
             trainer_args=args,
             results_file=args.out,
             experiment_name=args.exp,
             trial=args.trial)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp', type=str, default='experiment',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed')
    parser.add_argument('--trial', type=int, default=0,
                        help='--trial number - affects seed')
    parser.add_argument('--out', type=str, default='results.csv',
                        help='file to save test results')
    main(parser)
