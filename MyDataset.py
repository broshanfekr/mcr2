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
    def __init__(self, imgs, labels):
        self.images = imgs
        self.targets = labels

        BICUBIC = InterpolationMode.BICUBIC
        self.transform = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        tmp_img = self.images[idx]
        tmp_label = self.targets[idx]
        return tmp_img, tmp_label


def load_dataset(data_name, train=True, path="./data"):
    if data_name == "sampled_cifar10":
        file_path = os.path.join(path, "", "cifar10_5000samples.pckl")
        imgs, labels, X = load_var(file_path)
        trainset = MyCustomDataset(imgs=imgs, labels=labels)

    return trainset