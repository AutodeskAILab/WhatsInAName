from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl

import abcpartnames.transforms.hf
from abcpartnames.datasets.ABCTextDataset import ABCPairsDataModule
from abcpartnames.models.mlp import MLP
from abcpartnames.scripts.evaluate_model import evaluate


def main(parser):
    parser = MLP.add_model_specific_args(parser)
    parser = ABCPairsDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(seed=args.seed + args.trial)

    encoder = abcpartnames.transforms.hf.MaskedLMEncoder(model=args.model,
                                                         device='cuda:0' if args.accelerator == 'gpu' else 'cpu')

    evaluate(model=MLP(args, encoder=encoder),
             datamodule=ABCPairsDataModule(split='validation', **args.__dict__),
             trainer_args=args,
             results_file=args.out,
             experiment_name=args.exp,
             trial=args.trial)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp', type=str, default='experiment',
                        help='experiment name')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                        help='hugging face transformer model for embeddings')
    parser.add_argument('--seed', type=int, default=9876,
                        help='random seed')
    parser.add_argument('--trial', type=int, default=0,
                        help='--trial number - affects seed')
    parser.add_argument('--out', type=str, default='results.csv',
                        help='file to save test results')
    main(parser)
