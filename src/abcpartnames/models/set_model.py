import os
from argparse import ArgumentParser
from typing import Any, Optional, List

import pytorch_lightning as pl
import torch.cuda
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Adam

from abcpartnames.models.set_transformer import SetTransformer


def stack_or_unsqueeze(x: List[torch.Tensor]):
    """
    I have a list of one or more tensors.   I would 
    like to stack the tensors if there are more than 1
    of them.  If there is only one then I want to 
    unsqueeze the tensor in dimension 0.  i.e.
    like a stack of 1
    """
    if len(x) > 1:
        return torch.stack(x)
    return torch.unsqueeze(x, dim=0)


class SetModuleBase(pl.LightningModule):
    def __init__(self, hparams, encoder=None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.enc = encoder
        self.set_transformer = SetTransformer(dim_input=self.hparams.dim_input,
                                              num_outputs=1,
                                              dim_output=self.hparams.dim_out,
                                              num_inds=self.hparams.num_inds,
                                              dim_hidden=self.hparams.dim_hidden,
                                              num_heads=self.hparams.num_heads,
                                              ln=self.hparams.ln)
        self.example_input_array = [
                                       torch.randn([8, self.hparams.dim_input]),
                                       torch.randn([10, self.hparams.dim_input])
                                   ],

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group('SetModule')
        parser.add_argument('--num_inds', type=int, default=32,
                            help='number inducing points')
        parser.add_argument('--dim_input', type=int, default=768,
                            help='input dimensions to set transformer')
        parser.add_argument('--dim_hidden', type=int, default=512,
                            help='set transformer hidden units size')
        parser.add_argument('--dim_out', type=int, default=768,
                            help='set transformer output dimensions')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='number of attention heads')
        parser.add_argument('--ln', action='store_true', default=False,
                            help='use layer norm')
        parser.add_argument('--batch_size', type=int, default=512,
                            help='batch size')
        parser.add_argument('--bin_width', type=int, default=128,
                            help='width of bins to use in batched masking')
        return parent_parser

    def forward(self, X) -> Any:
        padded, mask = SetTransformer.pad_and_mask(X)
        out = self.set_transformer(padded, mask=mask).squeeze(1)
        return out

    def group_batch_forward(self, embs: List[torch.Tensor]):
        """
        Perform forward pass on batch of input embeddings and targets.

        Group assembly part embeddings according to the number of parts in each assembly. This is to improve memory
        efficiency by reducing padding. Groups based on bin size given by '--bin_width' param.

        The original order of the embs list is reinstated after the batching

        :param embs: Shape is N x P_i x 768 where N is no. of assemblies and P_i is number of parts in assembly i.
        """
        preds_ = []
        binned_embs = {}
        binned_indices = {}
        for index, emb in enumerate(embs):
            bin = emb.shape[0] // self.hparams.bin_width
            try:
                binned_embs[bin].append(emb)
                binned_indices[bin].append(index)
            except KeyError:
                binned_embs[bin] = [emb]
                binned_indices[bin] = [index]
        permutation = []
        for bin, embs in binned_embs.items():
            preds_.append(self.forward(embs))
            permutation.extend(binned_indices[bin])
        preds_ = torch.cat(preds_)

        # Use the permutation to reinstate the original ordering
        permutation = torch.tensor(permutation, dtype=torch.long)
        reverse_permutation = torch.argsort(permutation)
        preds_ = preds_[reverse_permutation]
        return preds_

    def embed_parts(self, chosen_parts: List[List[str]], names: List[str]):
        """
        Use the provided encoder (self.enc) to embed all inputs and target output.

        :param chosen_parts: selections of chosen parts for input
        :param names: names of each assembly
        :return: embs - Where embs has shape N x P_i x 768 where N is no. of
        assemblies and P_i is number of parts in assembly i
        """
        embs = [torch.stack([
            self.enc(x).to(self.device) for x in parts
        ]) for parts in chosen_parts]
        return embs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams,
                                    {'hp/test_acc': 0.,
                                     'hp/test_loss': 0.})


class SetModuleNamesWithParts(SetModuleBase):
    """
    This set module covers the experiments

    - Given the names of all the parts in the document, predict the 
      name of the document

    - Given the names of all but one of the parts document, predict
      the name of the missing part
    """

    def __init__(self, hparams, encoder=None) -> None:
        super().__init__(hparams, encoder)

    def embed_targets(self, targets: List[str]):
        """
        Embed the target strings

        :param targets: A list of target strings for each document in the batch
        :return: Tensor shape (batch_size, embedding_size)
        """
        return torch.stack([self.enc(t).to(self.device) for t in targets])

    def shared_step(self, batch):
        names, parts, targets = batch
        embs = self.embed_parts(parts, names)
        y = self.embed_targets(targets)
        preds = self.group_batch_forward(embs)
        return parts, preds, targets, y

    def training_step(self, batch, batch_id) -> STEP_OUTPUT:
        _, preds, _, y = self.shared_step(batch)
        loss = torch.nn.functional.mse_loss(preds, y)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        _, preds, targets, y = self.shared_step(batch)

        dist_matrix = torch.cosine_similarity(preds[:, :, None], y.T[None, :, :])
        pred_idxs = torch.argmax(dist_matrix, -1)

        correct = torch.sum(torch.arange(len(targets), device=self.device) == pred_idxs)
        acc = correct / len(targets)
        loss = torch.nn.functional.mse_loss(preds, y)
        self.log_dict({
            'val/acc': acc,
            'val/loss': loss
        }, batch_size=self.hparams.batch_size)
        return {'acc': acc.detach(), 'loss': loss.detach()}

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        input_parts, preds, targets, y = self.shared_step(batch)

        dist_matrix = torch.cosine_similarity(preds[:, :, None], y.T[None, :, :])
        pred_idxs = torch.argmax(dist_matrix, -1)

        correct = torch.sum(torch.arange(len(targets), device=self.device) == pred_idxs)
        acc = correct / len(targets)
        loss = torch.nn.functional.mse_loss(preds, y)
        self.log_dict({
            'hp/test_acc': acc,
            'hp/test_loss': loss
        }, batch_size=self.hparams.batch_size)

        tb = self.logger.experiment
        sample_path = f'{tb.log_dir}/samples.csv'
        if not os.path.exists(sample_path):
            with open(sample_path, 'w') as f:
                f.write('target\tprediction\tparts\n')
                for t, i, p in zip(targets, pred_idxs, input_parts):
                    f.write(f'{t}\t{targets[i]}\t{p}\n')
        with open(sample_path, 'a') as f:
            for t, i, p in zip(targets, pred_idxs, input_parts):
                f.write(f'{t}\t{targets[i]}\t{p}\n')

        return {'acc': acc.detach(), 'loss': loss.detach()}
