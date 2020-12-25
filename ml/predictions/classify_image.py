from ml.training.models import CassavaLite
from ml.training.config import CLASS_CATEGORIES, MODEL_NAME, IMG_SIZE

import torch
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import cv2

import os
import numpy as np


class ImageClassifier:
    def __init__(self):
        self.classifier = CassavaLite()
        model_path = os.path.join('ml', 'trained_model', MODEL_NAME)
        self.classifier = self.classifier.load_from_checkpoint(
            checkpoint_path=model_path)
        self.classifier = self.classifier.to('cpu')
        self.classifier.eval()
        self.classifier.freeze()

    def predict(self, image):
        test_transform = albu.Compose([
            albu.RandomResizedCrop(IMG_SIZE, IMG_SIZE),
            albu.Transpose(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

        # x = cv2.imread(image, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(image)
        image = test_transform(image=x)['image']
        # image = np.expand_dims(image, axis=0)
        image = image.unsqueeze(0)

        output = self.classifier(image)
        class_idx = torch.argmax(output, dim=1)

        return CLASS_CATEGORIES[class_idx]
