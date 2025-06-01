#imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from timm import create_model, list_models
import logging
from sklearn.model_selection import train_test_split

#globals
data_directory = "small"
catagories = ["Academic_Art", "Art_Nouveau", "Baroque", "Expressionism", "Japanese_Art", "Neoclassicism", "Primitivism", "Realism", "Renaissance", "Rococo", "Romanticism", "Symbolism", "Western_Medieval"]
random_state = 99
class_numbers = 13
epochs = 10

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

def prepare_data(cat):
    train_data = datasets.ImageFolder(data_directory + "/" + cat, transform=data_transforms)


    return DataLoader(train_data, batch_size=32, shuffle=True)

#evaluation functions
def save_image(loader):
    imgs, labs = next(iter(loader))
    to_pil = transforms.ToPILImage()
    for i, img in enumerate(imgs):
        pil_image = to_pil(img)
        pil_image.save(f"{cat}_{i}.jpg")

    
    

if __name__ == "__main__":
    for cat in catagories:
        train_loader = prepare_data(cat)
        save_image(train_loader)
