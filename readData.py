# %%

import os, sys, glob, shutil, json
import cv2

from PIL import Image

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


# %%

class SVHDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # original SVHN's char--'10' correlated to class '0'
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)


# %%


train_path = glob.glob('input/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('input/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
    SVHDataset(train_path, train_label, transforms.Compose([
        # resize the fixed size
        transforms.Resize((64, 128)),

        # random crop
        transforms.RandomCrop((60,120)),

        # change the color randomly
        transforms.ColorJitter(0.2, 0.2, 0.2),

        # rotate randomly
        transforms.RandomRotation(5),

        # transform the img to tensor of 'Pytorch'
        transforms.ToTensor(),

        # normalize the img
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=10,
    shuffle=False,
    num_workers=10,

)

for data in train_loader:
    break


