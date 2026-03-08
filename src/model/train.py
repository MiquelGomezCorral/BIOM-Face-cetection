import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.config import Configuration


def train_model(CONFIG: Configuration, data_module, model_module, ckpt_path: str = None):
    save_path = ckpt_path or CONFIG.best_cnn_model_path
    dirpath   = os.path.dirname(save_path)
    filename  = os.path.splitext(os.path.basename(save_path))[0]

    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=dirpath,
                filename=filename,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model_module, datamodule=data_module)
    trainer.test(model_module, datamodule=data_module)
