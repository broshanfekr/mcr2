import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import torchvision.transforms as T

from PIL import Image as im 


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


class MyCustomDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.images = imgs
        self.targets = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tmp_img = self.images[idx]
        tmp_label = self.targets[idx]
        
        if self.transform:
            tmp_img = self.transform(tmp_img)
        
        return tmp_img, tmp_label


def load_dataset(data_name, transform=None, path="./data"):
    if data_name == "sampled_cifar10":
        file_path = os.path.join(path, "", "cifar10_5000samples.pckl")
        imgs, labels, X = load_var(file_path)
        trainset = MyCustomDataset(imgs=imgs, labels=labels, transform=transform)

    return trainset