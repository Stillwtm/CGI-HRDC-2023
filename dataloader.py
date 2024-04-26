import os
import pandas as pd
import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from preprocess import preprocess_image

def random_rotation(image, max_angle=30):
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy.ndimage.rotate(image, angle, reshape=False, mode='nearest')

class MyDataset(Dataset):
    def __init__(self, image_dir, csv_file, img_size=512, transform=None):
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.csv_file.iloc[idx, 0])

        image = cv2.imread(image_path)
        image = preprocess_image(image)
        if self.transform is not None:
            image = self.transform(Image.fromarray(image))
        #     print("Transform")
        # else:
        #     print("No transform")
        # image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # image = image / 255.0
        # image = random_rotation(image, max_angle=10)
        # image = np.moveaxis(image, -1, 0)
        label = self.csv_file.iloc[idx, 1]

        return image, label


def create_dataloader(train_batch_size: int, val_batch_size: int, img_size: int = 512, 
                      use_full_data: bool = False, task: str = "task1"):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5024, 0.5013, 0.5009], std=[0.0007, 0.0008, 0.0009])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5024, 0.5013, 0.5009], std=[0.0007, 0.0008, 0.0009])
    ])

    dataset = MyDataset(f"./dataset/image_{task}", f"./dataset/label_{task}.csv", img_size=img_size, transform=train_transform)

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    if use_full_data:
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return train_loader, val_loader
