import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


def evaluate(model, datamodule, trainer_args, results_file, experiment_name, trial):
    trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        max_epochs=200,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir='lightning_logs',
                                 name=experiment_name,
                                 default_hp_metric=False),
        callbacks=[
            ModelCheckpoint(monitor='val_loss',
                            mode='min',
                            filename='best-{epoch:02d}-{step:05d}-{val_acc:.2f}-{val_loss:.2f}',
                            save_last=True),
            EarlyStopping(monitor='val_loss',
                          patience=5,
                          mode='min')
        ],
    )

    trainer.fit(model=model,
                datamodule=datamodule)

    result = trainer.test(model=model,
                          ckpt_path='best',
                          datamodule=datamodule)
    with open(results_file, 'a') as f:
        f.write(f"{experiment_name},{trial},{(result[0]['test_loss'])},{(result[0]['test_acc'])}\n")
