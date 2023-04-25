import json
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from distutils.util import strtobool
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.loggers import TensorBoardLogger

from abcpartnames.datasets.ABCTextDataset import (
    ABCNamesWithPartsDataModule
)
from abcpartnames.models.set_model import SetModuleNamesWithParts
from abcpartnames.transforms.hf import ToPreTrainedMaskedLMEmbedding
from abcpartnames.transforms.vectorizers import ToFastText, ToTechNet


def get_names_with_parts_datamodule(args):
    return ABCNamesWithPartsDataModule(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pred_names=args.pred_names,
        no_lower=args.no_lower,
        replace_=args.replace_
    )


def get_datamodule(args):
    """
    Get the train and test datamodules
    """
    return get_names_with_parts_datamodule(args)


def get_set_module(args, encoder):
    return SetModuleNamesWithParts(args, encoder=encoder)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trial', type=int, default=0,
                        help='trial (used for random seed)')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                        help='pre-trained language model')
    parser.add_argument('--exp', type=str, default='test',
                        help='experiment_name')
    parser.add_argument('--results', type=str, default='set_results.csv',
                        help='file to save results')
    parser.add_argument('--workers', type=int, default=8,
                        help='num workers for dataloaders')
    parser.add_argument('--stop_on', type=str, default='loss',
                        help='\'loss\' or \'acc\'')
    SetModuleNamesWithParts.add_argparse_args(parser)
    ToFastText.add_argparse_args(parser)
    ABCNamesWithPartsDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(9876 + args.trial)

    print('load dataset...')
    data_module = get_datamodule(args)

    print('create set model...')
    if args.model == 'fasttext':
        enc = ToFastText(args.gensim_path)
    else:
        if args.model == 'technet':
            model_supplier = lambda: ToTechNet()
        else:
            model_supplier = lambda: ToPreTrainedMaskedLMEmbedding(model=args.model,
                                                                   device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                                                   with_cache=True)

        model_options = {}
        enc = data_module.create_caching_encoder(model_name=args.model,
                                                 model_options=model_options,
                                                 model_supplier=model_supplier)
    lm = get_set_module(args, enc)

    logger = TensorBoardLogger(save_dir='lightning_logs',
                               name=args.exp,
                               default_hp_metric=False,
                               log_graph=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss' if args.stop_on == 'loss' else 'val/acc',
                                                       mode='min' if args.stop_on == 'loss' else 'max',
                                                       every_n_epochs=1)
    stopping_callback = pl.callbacks.EarlyStopping(monitor='val/loss' if args.stop_on == 'loss' else 'val/acc',
                                                   mode='min' if args.stop_on == 'loss' else 'max',
                                                   patience=40)

    trainer = pl.Trainer(max_epochs=200,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback, stopping_callback],
                         logger=logger,
                         accelerator='gpu',
                         gpus=1)

    print('fitting...')
    trainer.fit(model=lm, datamodule=data_module)

    result = trainer.test(model=lm,
                          ckpt_path='best',
                          datamodule=data_module)

    test_acc = result[0]['hp/test_acc']
    test_loss = result[0]['hp/test_loss']

    if not os.path.exists(args.results):
        os.makedirs(Path(args.results).parent, exist_ok=True)
        with open(args.results, 'w') as f:
            f.write('experiment;trial;args;test_loss;test_acc\n')

    with open(args.results, 'a') as f:
        f.write(f'{args.exp};{args.trial};{json.dumps(args.__dict__)};{test_loss};{test_acc}\n')
