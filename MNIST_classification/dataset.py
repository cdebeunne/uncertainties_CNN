import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms

class ImageDataset(Dataset):

    def __init__(self, im, ind_im, train=True):
        """
        :param list_im:  List of image path
        :param ind_im: List of class Indice
        :param train: Bool, True if training mod
        """
        self.im = im
        self.im_ind = ind_im
        self.train = train

    def __len__(self):
        return len(self.im)

    def __getitem__(self, index):
        # Get image & convert it in torch tensor
        img = self.im[index]
        x = img

        # Get class of the image (Groundtruth)
        y = self.im_ind[index]

        # Transform image with torchvision functions (if train mod)
        toPil = transforms.ToPILImage()
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(0.5), # Flip the data horizontally
            transforms.RandomRotation(2.5),
            transforms.ToTensor(),
            ])
        transformTensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.train:
          x = transform(toPil(x))
        else:
          x = transformTensor(toPil(x))
            
        
        return x,y