import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from time import time

import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec


class LossLogger(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(self):
        self.epoch = 1
        self._losses = [0.]
        self.losses = []
        self.models = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss - self._losses[-1])
        print(f'  Loss: {loss - self._losses[-1]}')
        self._losses.append(loss)

        version = time()
        model_file = f'gensim_logs/{version}.gensim'
        model.save(model_file)
        self.models.append(model_file)
        self.epoch += 1


def main(args):
    print(args.__dict__)
    os.makedirs('gensim_logs', exist_ok=True)

    if args.method == 'word2vec':
        results_file = 'gensim_logs/results.csv'
        logger = LossLogger()

        Word2Vec(corpus_file=args.corpus_file,
                 sg=args.sg,
                 vector_size=args.vector_size,
                 window=args.window,
                 negative=args.negative,
                 workers=args.workers,
                 compute_loss=True,
                 callbacks=[logger],
                 epochs=15)

        df = pd.DataFrame({
            'loss': logger.losses,
            'model': logger.models,
            'args': json.dumps(args.__dict__)
        })
    else:
        results_file = 'gensim_logs/results_fasttext.csv'
        ft = FastText(corpus_file=args.corpus_file,
                      sg=args.sg,
                      vector_size=args.vector_size,
                      window=args.window,
                      negative=args.negative,
                      workers=args.workers)
        version = time()
        model_file = f'gensim_logs/{version}.gensim'
        ft.save(model_file)

        df = pd.DataFrame({
            'model': [model_file],
            'args': [json.dumps(args.__dict__)]
        })

    if os.path.exists(results_file):
        df.to_csv(results_file,
                  index=False,
                  header=False,
                  mode='a')
    else:
        df.to_csv(results_file,
                  index=False,
                  header=True,
                  mode='w')


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', type=str, default='fasttext',
                        help='\'word2vec\' or \'fasttext\'')
    parser.add_argument('--corpus_file', type=str, default='data/abc/train_corpus_complete.txt',
                        help='train lines input txt file')
    parser.add_argument('--vector_size', type=int, default=100,
                        help='size of embeddings')
    parser.add_argument('--window', type=int, default=5,
                        help='max window between current word and predicted')
    parser.add_argument('--sg', type=int, default=0,
                        help='1 for skip-gram, 0 for CBOW')
    parser.add_argument('--negative', type=int, default=5,
                        help='number of negative sampling')
    parser.add_argument('--workers', type=int, default=8,
                        help='num workers')
    main(parser.parse_args())
