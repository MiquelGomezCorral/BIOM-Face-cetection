import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.config import Configuration


def train_model(CONFIG: Configuration, data_module, model_module):
    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="face-{epoch:02d}-{val_loss:.4f}",
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model_module, datamodule=data_module)
    trainer.test(model_module, datamodule=data_module)
