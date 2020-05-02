import os
import time
import sys

import torch
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms


class AugmentLoader:
    def __init__(self, 
              dataset, 
              batch_size,
              transforms=transforms.ToTensor(),
              num_aug=None, 
              shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transforms = transforms
        self.num_aug = num_aug
        self.shuffle = shuffle
        self.size = len(self.dataset.targets)
        self.sample_indices = self.reset_index()
    
    def __iter__(self):
        return _Iter(self)
    
    def reset_index(self):
        if self.shuffle:
            return np.random.choice(self.size, self.size, replace=False).tolist()
        else:
            return np.arange(self.size).tolist()
    
    def apply_augments(self, sample):
        if self.num_aug is None:
            return self.transforms(sample).unsqueeze(0)
        batch_imgs = [transforms.ToTensor()(sample).unsqueeze(0)]
        for _ in range(self.num_aug):
            transformed = self.transforms(sample)
            batch_imgs.append(transformed.unsqueeze(0))
        return torch.cat(batch_imgs, axis=0)
    

class _Iter():
    def __init__(self, loader):
        self.loader = loader
    def __next__(self):
        if len(self.loader.sample_indices) < self.loader.batch_size:
            self.loader.sample_indices = self.loader.reset_index()
            raise StopIteration
        batch_imgs = []
        batch_lbls = []
        batch_idx = []
        for img_idx in range(self.loader.batch_size):
            sample_index = self.loader.sample_indices.pop(0)
            img, label = self.loader.dataset[sample_index]
            img_augments = self.loader.apply_augments(img)
            batch_imgs.append(img_augments)
            num_aug = self.loader.num_aug or 0
            batch_lbls.append(np.repeat(label, num_aug+1))
            batch_idx.append(np.repeat(img_idx, num_aug+1))
        batch_imgs = torch.cat(batch_imgs, axis=0).float()
        batch_lbls = torch.from_numpy(np.hstack(batch_lbls))
        batch_idx = torch.from_numpy(np.hstack(batch_idx))
        return (batch_imgs,
                batch_lbls,
                batch_idx)