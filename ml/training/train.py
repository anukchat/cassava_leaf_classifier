from ml.training.data import CassavaDataModule, CassavaDataModule
from ml.training.models import CassavaLite
from ml.training import config

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(42)

# Callbacks
model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                   verbose=True,
                                   filename="{epoch}_{val_loss:.4f}")
early_stopping = EarlyStopping('val_loss', patience=4)


dm = CassavaDataModule()
cassava_model = CassavaLite()
trainer = pl.Trainer(gpus=-1, max_epochs=12,
                     callbacks=[model_checkpoint, early_stopping])
trainer.fit(cassava_model, dm)

# manually you can save best checkpoints -
# trainer.save_checkpoint("cassava_efficient_net.ckpt")
torch.save(dm.save_dict(), config.MODEL_NAME)
