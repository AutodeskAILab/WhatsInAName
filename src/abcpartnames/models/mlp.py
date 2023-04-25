from typing import Optional, Union, List, Any

import torch.nn
from pl_bolts.metrics import accuracy
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from sklearn.metrics import ConfusionMatrixDisplay
from torch import optim
from torchmetrics.functional import accuracy


class MLP(LightningModule):
    def __init__(self, args, encoder=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.save_hyperparameters(args)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.hparams.embedding_size * 2, out_features=self.hparams.hidden_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.hparams.hidden_dims, out_features=self.hparams.num_classes)
        )
        self.val_acc_best = 0.

    def forward(self, a, b, **kwargs) -> Any:
        if self.encoder:
            with torch.no_grad():
                a = self.encoder(a)
                b = self.encoder(b)
        x = torch.cat([a, b], dim=-1)
        return self.net(x)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {
            'hp/val_acc_best': 0,
            'hp/test_acc': 0
        })

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        t1, t2, y = batch
        logits = self.forward(t1, t2)
        loss = torch.nn.functional.cross_entropy(input=logits, target=y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        t1, t2, y = batch
        logits = self.forward(t1, t2)
        loss = torch.nn.functional.cross_entropy(input=logits, target=y)
        self.log('val_loss', loss)
        preds = logits.argmax(dim=-1)
        acc = accuracy(preds, y)
        return acc

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        acc = torch.stack(outputs).mean()
        if acc > self.val_acc_best:
            self.val_acc_best = acc
            self.log('hp/val_acc_best', acc)
        self.log('val_acc', acc)
        return acc

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        t1, t2, y = batch
        logits = self.forward(t1, t2)
        loss = torch.nn.functional.cross_entropy(input=logits, target=y)
        self.log('test_loss', loss)
        preds = logits.argmax(dim=-1)
        acc = accuracy(preds, y)
        return {'acc': acc, 'preds': preds, 'loss': loss, 'labels': y, 't1': t1, 't2': t2}

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        acc = []
        preds = []
        loss = []
        labels = []
        t1 = []
        t2 = []
        for output in outputs:
            acc.append(output['acc'])
            preds.append(output['preds'])
            loss.append(output['loss'])
            labels.append(output['labels'])
            t1 += output['t1']
            t2 += output['t2']
        acc = torch.stack(acc).mean()
        loss = torch.stack(loss).mean()
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        self.log('test_acc', acc)
        self.log('test_loss', loss)

        self.log('hp/test_acc', acc)

        cm = ConfusionMatrixDisplay.from_predictions(preds.detach().cpu(), labels.detach().cpu())
        tb = self.logger.experiment
        tb.add_figure('confusion_matrix', cm.figure_)

        tp = torch.arange(len(labels))[(preds == 1) & (labels == 1)]
        fp = torch.arange(len(labels))[(preds == 1) & (labels == 0)]
        tn = torch.arange(len(labels))[(preds == 0) & (labels == 0)]
        fn = torch.arange(len(labels))[(preds == 0) & (labels == 1)]

        with open(f'{tb.log_dir}/samples.txt', 'w') as f:
            f.write('===True Positives===\n')
            for i in range(min(10, len(tp))):
                f.write(f'{t1[tp[i]]},{t2[tp[i]]}\n')

            f.write('\n===False Positives===\n')
            for i in range(min(10, len(fp))):
                f.write(f'{t1[fp[i]]},{t2[fp[i]]}\n')

            f.write('\n===True Negatives===\n')
            for i in range(min(10, len(tn))):
                f.write(f'{t1[tn[i]]},{t2[tn[i]]}\n')

            f.write('\n===False Negatives===\n')
            for i in range(min(10, len(fn))):
                f.write(f'{t1[fn[i]]},{t2[fn[i]]}\n')

        return acc

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        a, b, y, t1, t2 = batch
        logits = self.forward(a, b)
        preds = logits.argmax(dim=-1)
        return preds

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.hparams.weight_decay)
        return opt

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLP")
        parser.add_argument("--embedding_size", type=int, default=768)
        parser.add_argument('--hidden_dims', type=int, default=100)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--weight_decay', type=float, default=0.)
        return parent_parser


class SparseMLP(MLP):
    def forward(self, a, b, **kwargs) -> Any:
        a = a.float().to_dense().squeeze()
        b = b.float().to_dense().squeeze()
        return super().forward(a, b, **kwargs)
