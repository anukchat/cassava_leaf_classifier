import cv2
import os
import pandas as pd
import albumentations as albu
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from ml.training import config


class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_id'])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if(self.transforms):
            x = self.transforms(image=x)['image']

        if(self.train):
            y = self.df.iloc[index]['label']
            return {
                "x": x,
                "y": y,
            }
        else:
            return {"x": x}

    def __len__(self):
        return len(self.df)


class CassavaDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_transform = albu.Compose([
            albu.RandomResizedCrop(config.IMG_SIZE, config.IMG_SIZE, p=1.0),
            albu.Transpose(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            albu.CoarseDropout(p=0.5),
            albu.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

        self.valid_transform = albu.Compose([
            albu.CenterCrop(config.IMG_SIZE, config.IMG_SIZE, p=1.),
            albu.Resize(config.IMG_SIZE, config.IMG_SIZE),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

    def prepare_data(self):
        # prepare_data is called only once on 1- GPU in a distributed computing
        df = pd.read_csv(config.TRAIN_CSV)
        df["kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        stratify = StratifiedKFold(n_splits=5)
        for i, (t_idx, v_idx) in enumerate(stratify.split(X=df.image_id.values, y=df.label.values)):
            df.loc[v_idx, "kfold"] = i
            df.to_csv("train_folds.csv", index=False)

    def setup(self, stage=None):
        dfx = pd.read_csv("train_folds.csv")
        train = dfx.loc[dfx["kfold"] != 1]
        val = dfx.loc[dfx["kfold"] == 1]

        self.train_dataset = CassavaDataset(
            train,
            config.TRAIN_IMAGES_DIR,
            train=True,
            transforms=self.train_transform)

        self.valid_dataset = CassavaDataset(
            val,
            config.TRAIN_IMAGES_DIR,
            train=True,
            transforms=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=4)
