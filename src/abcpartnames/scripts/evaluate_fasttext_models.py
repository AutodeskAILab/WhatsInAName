from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
from ast import literal_eval

import pandas as pd
import pytorch_lightning

from abcpartnames.datasets.ABCTextDataset import ABCPairsDataModule
from abcpartnames.models.mlp import MLP
from abcpartnames.scripts.evaluate_model import evaluate
from abcpartnames.transforms.vectorizers import ToFastText

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser = ABCPairsDataModule.add_argparse_args(parser)
    trainer_args = parser.parse_args()
    trainer_args.accelerator = 'gpu'
    trainer_args.gpus = 1

    df = pd.read_csv('gensim_logs/results_fasttext.csv')
    for model, args in df[['model', 'args']].values:
        group_args = literal_eval(args)
        model_num = model.rpartition('/')[2]
        group_args['model_num'] = model_num

        mlp_args = Namespace(embedding_size=group_args['vector_size'],
                             hidden_dims=100,
                             num_classes=2,
                             weight_decay=0.,
                             fasttext=model,
                             **group_args)

        transform = ToFastText(gensim_path=model)
        datamodule = ABCPairsDataModule(split='validation',
                                        transform=transform,
                                        **trainer_args.__dict__)

        for i in range(10):
            pytorch_lightning.seed_everything(9876 + i)
            experiment_name = f'fasttext_{group_args["vector_size"]}_win_{group_args["window"]}'
            experiment_name += f'_sg_{group_args["sg"]}_model_{group_args["model_num"]}'
            evaluate(model=MLP(mlp_args),
                     datamodule=datamodule,
                     trainer_args=trainer_args,
                     results_file='results.csv',
                     experiment_name=experiment_name,
                     trial=i)
