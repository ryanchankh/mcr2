import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cluster import ElasticNetSubspaceClustering, clustering_accuracy
import utils


def load_architectures(name, dim):
    """Returns a network architecture.
    
    Parameters:
        name (str): name of the architecture
        dim (int): feature dimension of vector presentation
    
    Returns:
        net (torch.nn.Module)
        
    """
    _name = name.lower()
    if _name == "resnet18":
        from architectures.resnet_cifar import ResNet18
        net = ResNet18(dim)
    elif _name == "resnet18ctrl":
        from architectures.resnet_cifar import ResNet18Control
        net = ResNet18Control(dim)
    elif _name == "resnet18stl":
        from architectures.resnet_stl import ResNet18STL
        net = ResNet18STL(dim)
    elif _name == "vgg11":
        from architectures.vgg_cifar import VGG11
        net = VGG11(dim)
    elif _name == "resnext29_2x64d":
        from architectures.resnext_cifar import ResNeXt29_2x64d
        net = ResNeXt29_2x64d(dim)
    elif _name == "resnext29_4x64d":
        from architectures.resnext_cifar import ResNeXt29_4x64d
        net = ResNeXt29_4x64d(dim)
    elif _name == "resnext29_8x64d":
        from architectures.resnext_cifar import ResNeXt29_8x64d
        net = ResNeXt29_8x64d(dim)
    elif _name == "resnext29_32x4d":
        from architectures.resnext_cifar import ResNeXt29_32x4d
        net = ResNeXt29_32x4d(dim)
    elif _name == "resnet10mnist":
        from architectures.resnet_mnist import ResNet10MNIST
        net = ResNet10MNIST(dim)
    else:
        raise NameError("{} not found in architectures.".format(name))
    net = torch.nn.DataParallel(net).cuda()
    return net


def load_trainset(name, transform=None, train=True, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        name (str): name of the dataset
        transform (torchvision.transform): transform to be applied
        train (bool): load trainset or testset
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    _name = name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "cifar10"), train=train,
                                                download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "cifar100coarse":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.targets = sparse2coarse(trainset.targets) 
        trainset.num_classes = 20
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root=os.path.join(path, "mnist"), train=train, 
                                              download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "stl10":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.data = np.concatenate([trainset.data, testset.data])
            trainset.labels = trainset.labels.tolist() + testset.labels.tolist()
            trainset.targets = trainset.labels
            return trainset
    elif _name == "stl10sup":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset
        else:
            trainset.targets = trainset.labels
            return trainset
    else:
        raise NameError("{} not found in trainset loader".format(name))
    return trainset


def load_transforms(name):
    """Load data transformations.
    
    Note:
        - Gaussian Blur is defined at the bottom of this file.
    
    """
    _name = name.lower()
    if _name == "default":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif _name == "cifar":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    elif _name == "mnist":
         transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]), 
                GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "stl10":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=9),
            transforms.ToTensor()])
    elif _name == "fashionmnist" or _name == "fmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "test":
        transform = transforms.ToTensor()
    else:
        raise NameError("{} not found in transform loader".format(name))
    return transform


def load_checkpoint(model_dir, epoch=None, eval_=False):
    """Load checkpoint from model directory. Checkpoints should be stored in 
    `model_dir/checkpoints/model-epochX.ckpt`, where `X` is the epoch number.
    
    Parameters:
        model_dir (str): path to model directory
        epoch (int): epoch number; set to None for last available epoch
        eval_ (bool): PyTorch evaluation mode. set to True for testing
        
    Returns:
        net (torch.nn.Module): PyTorch checkpoint at `epoch`
        epoch (int): epoch number
    
    """
    if epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
    ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
    params = utils.load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    net = load_architectures(params['arch'], params['fd'])
    net.load_state_dict(state_dict)
    del state_dict
    if eval_:
        net.eval()
    return net, epoch

    
def get_features(net, trainloader, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        batch_features = net(batch_imgs.cuda())
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)
    

def corrupt_labels(mode="default"):
    """Returns higher corder function"""
    if mode == "default":
        from corrupt import default_corrupt
        return default_corrupt
    elif mode == "asymmetric_noise":
        from corrupt import asymmetric_noise
        return asymmetric_noise
    elif mode == "noisify_pairflip":
        from corrupt import noisify_pairflip
        return noisify_pairflip
    elif mode == "noisify_multiclass_symmetric":
        from corrupt import noisify_multiclass_symmetric
        return noisify_multiclass_symmetric



def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi


def membership_to_label(membership):
    """Turn a membership matrix into a list of labels."""
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i, i])
    return labels

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


## Additional Augmentations
class GaussianBlur():
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]
