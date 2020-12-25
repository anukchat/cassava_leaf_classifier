import sys
import pandas as pd
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score

import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnext50_32x4d
from torch.utils.data import Dataset, DataLoader
from ml.training import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from efficientnet_pytorch import EfficientNet


class CassavaLite(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.efficient_net = EfficientNet.from_name('efficientnet-b5')
        # self.efficient_net.load_state_dict(torch.load(config.PRETRAINED_PATH))
        self.efficient_net = EfficientNet.from_pretrained(
            'efficientnet-b5', num_classes=config.CLASSES)
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, config.CLASSES)

    def forward(self, x):
        out = self.efficient_net(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-4, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
