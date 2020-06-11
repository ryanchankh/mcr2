import os
import time
import sys

import torch
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms


class AugmentLoader:
    """Dataloader that includes augmentation functionality.
    
    Parameters:
        dataset (torch.data.dataset): trainset or testset PyTorch object
        batch_size (int): the size of each batch, including augmentations
        sampler (str): choice of sampler ('balance' or 'random')
            - 'balance': samples data such that each class has the same number of samples
            - 'random': samples data randomly
        transforms (torchvision.transforms): Transformations applied to each augmentation
        num_aug (int): number of augmentation for each image in a batch
        shuffle (bool): shuffle data
        
    Attributes:
        dataset (torch.data.dataset): trainset or testset PyTorch object
        batch_size (int): the size of each batch, including augmentations
        transforms (torchvision.transforms): Transformations applied to each augmentation
        num_aug (int): number of augmentation for each image in a batch
        shuffle (bool): shuffle data
        size (int): number of samples in dataset
        sample_indices (np.ndarray): indices for sampling

    Notes:
        - number of augmetations and batch size are used to calculate the number of original 
        images used in a batch
        - if num_aug = 0, then this dataloader is the same as an PyTorch dataloader, with 
        the number of original images equal to the batch size, and each image is transformed 
        using transforms from object argument.
        - Auygmentloder first samples from the dataset num_img of images, then apply augmentation 
        to all images. The first augmentation is always the identity transform. 

    """
    def __init__(self, 
              dataset, 
              batch_size,
              sampler="random",
              transforms=transforms.ToTensor(),
              num_aug=0, 
              shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.transforms = transforms
        self.sampler = sampler
        self.num_aug = num_aug
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.sampler == "balance":
            sampler = BalanceSampler(self.dataset)
            num_img = self.batch_size // self.num_aug
            return _Iter(self, sampler, num_img, self.num_aug)
        elif self.sampler == "random":
            size = len(self.dataset.targets) // self.batch_size * self.batch_size
            sampler = RandomSampler(self.dataset, size, shuffle=self.shuffle)
            num_img = self.batch_size // self.num_aug
            return _Iter(self, sampler, num_img, self.num_aug)
        else:
            raise NameError(f"sampler {self.sampler} not found.")

    def update_labels(self, targets):
        self.dataset.targets = targets

    def apply_augments(self, sample):
        if self.num_aug is None:
            return self.transforms(sample).unsqueeze(0)
        batch_imgs = [transforms.ToTensor()(sample).unsqueeze(0)]
        for _ in range(self.num_aug-1):
            transformed = self.transforms(sample)
            batch_imgs.append(transformed.unsqueeze(0))
        return torch.cat(batch_imgs, axis=0)
    

class _Iter():
    def __init__(self, loader, sampler, num_img, num_aug, size=None):
        self.loader = loader
        self.sampler = sampler
        self.num_img = num_img
        self.num_aug = num_aug
        self.size = size

    def __next__(self):
        if self.sampler.stop():
            raise StopIteration
        batch_imgs = []
        batch_lbls = []
        batch_idx = []
        sampled_imgs, sampled_lbls = self.sampler.sample(self.num_img)
        for i in range(self.num_img):
            img_augments = self.loader.apply_augments(sampled_imgs[i])
            batch_imgs.append(img_augments)
            batch_lbls.append(np.repeat(sampled_lbls[i], self.num_aug))
            batch_idx.append(np.repeat(i, self.num_aug))
        batch_imgs = torch.cat(batch_imgs, axis=0).float()
        batch_lbls = torch.from_numpy(np.hstack(batch_lbls))
        batch_idx = torch.from_numpy(np.hstack(batch_idx))
        return (batch_imgs,
                batch_lbls,
                batch_idx)


class BalanceSampler():
    """Samples data such that each class has the same number of samples. Performs sampling 
    by first sorting data then unfiormly sample from batch with replacement."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(self.dataset.targets)
        self.num_classes = np.max(self.dataset.targets) + 1
        self.num_sampled = 0
        self.sort()


    def sort(self):
        sorted_data = [[] for _ in range(self.num_classes)]
        for i, lbl in enumerate(self.dataset.targets):
            sorted_data[lbl].append(self.dataset[i][0])
        self.sorted_data = sorted_data
        self.sorted_labels = [np.repeat(i, len(sorted_data[i])) for i in range(self.num_classes)]


    def sample(self, num_imgs):
        num_imgs_per_class = num_imgs // self.num_classes
        assert num_imgs_per_class * self.num_classes == num_imgs, 'cannot sample uniformly'

        batch_imgs, batch_lbls = [], []
        for c in range(self.num_classes):
            img_c, lbl_c = self.sorted_data[c], self.sorted_labels[c]
            sample_indices = np.random.choice(len(img_c), num_imgs_per_class)
            for i in sample_indices:
                batch_imgs.append(img_c[i])
                batch_lbls.append(lbl_c[i])
        self.increment_step(num_imgs)
        return batch_imgs, batch_lbls


    def increment_step(self, num_imgs):
        self.num_sampled += num_imgs


    def stop(self):
        if self.num_sampled < self.size:
            return False
        return True


class RandomSampler():
    """Samples data randomly. Sampler initializes sample indices when Sampler is instantiated.
    Sample indices are shuffled if shuffle option is True. Performs sampling by popping off 
    first index each time."""
    def __init__(self, dataset, size, shuffle=False):
        self.dataset = dataset
        self.size = size
        self.shuffle = shuffle
        self.num_sampled = 0
        self.sample_indices = self.reset_index()

    def reset_index(self):
        if self.shuffle:
            return np.random.choice(len(self.dataset.targets), self.size, replace=False).tolist()
        else:
            return np.arange(self.size).tolist()
        
    def sample(self, num_img):
        indices = [self.sample_indices.pop(0) for _ in range(num_img)]
        batch_imgs, batch_lbls = [], []
        for i in indices:
            img, lbl = self.dataset[i]
            batch_imgs.append(img)
            batch_lbls.append(lbl)
        self.increment_step(num_img)
        return batch_imgs, batch_lbls

    def increment_step(self, num_img):
        self.num_sampled += num_img

    def stop(self):
        if self.num_sampled < self.size:
            return False
        return True