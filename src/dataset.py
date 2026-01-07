import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd


ALL_LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
'Hernia', 'Infiltration', 'Mass', 'Nodule',
'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def encode(self, labels):
        labels = labels.split('|')
        return [1 if l in labels else 0 for l in ALL_LABELS] # returns a list of length 14 with 0 and 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).float()

        label = torch.tensor(self.encode(row['Finding Labels'])).float() # tensor of length 14 with 0. and 1.

        if self.transform:
            img = self.transform(img)

        return img, label
    
    def test(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image Index'])

        print(img_path)