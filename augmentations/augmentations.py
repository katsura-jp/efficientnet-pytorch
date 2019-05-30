import torch
import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensor

class TrainAugment(object):
    def __init__(self):
        self.Compose = albu.Compose([
            albu.PadIfNeeded(min_height=40, min_width=40, border_mode=0, value=[0,0,0], always_apply=True),
            albu.Cutout(num_holes=3, max_h_size=4, max_w_size=4, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.RandomCrop(height=32, width=32, always_apply=True),
            albu.ToFloat(max_value=None, always_apply=True),
            ToTensor(normalize={'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]})
        ])
    def __call__(self, image):
        transformed = self.Compose(image=image)
        image = transformed['image']
        return image


class TestAugment(object):
    def __init__(self):
        self.Compose = albu.Compose([
            albu.HorizontalFlip(p=0),
            albu.ToFloat(max_value=None, always_apply=True),
            ToTensor(normalize={'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]})
        ])
    def __call__(self, image):
        transformed = self.Compose(image=image)
        image = transformed['image']
        return image