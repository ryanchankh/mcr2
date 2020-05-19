import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from clustering import ElasticNetSubspaceClustering, clustering_accuracy
import utils


def load_architectures(name, dim):
    _name = name.lower()
    if _name == "resnet18":
        from architectures.resnet_cifar import ResNet18
        net = ResNet18(dim)
    elif _name == "resnet34":
        from architectures.resnet_cifar import ResNet34
        net = ResNet34(dim)
    elif _name == "resnet50":
        from architectures.resnet_cifar import ResNet50
        net = ResNet50(dim)
    elif _name == "resnet101":
        from architectures.resnet_cifar import ResNet101
        net = ResNet101(dim)
    elif _name == "resnet152":
        from architectures.resnet_cifar import ResNet152
        net = ResNet152(dim)
    elif _name == "resnet18mod":
        from architectures.resnet_cifar import ResNet18Mod
        net = ResNet18Mod(dim)
    elif _name == "resnet18old":
        from architectures.resnet_cifar import ResNet18Old
        net = ResNet18Old(dim) 
    else:
        raise NameError("{} not found in archiectures.".format(name))
    
    # return net.cuda()
    return torch.nn.DataParallel(net).cuda()


def load_trainset(name, transform=None, train=True):
    _name = name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=train,
                                                download=True, transform=transform)
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=train,
                                                 download=True, transform=transform)
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root="./data/mnist/", train=train, 
                                              download=True, transform=transform)
    elif _name =="fashion_mnist":
        trainset = torchvision.datasets.FashionMNIST(root="./data/fashion_mnist/", train=train, 
                                              download=True, transform=transform) 
    elif _name =="usps":
        trainset = torchvision.datasets.USPS(root="./data/usps/", train=train, 
                                              download=True, transform=transform) 
    else:
        raise NameError("{} not found in trainset loader".format(name))
    return trainset


def load_transforms(name):
    _name = name.lower()
    if _name == "default":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif _name == "simclr":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    elif _name == "augment":
         transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=(0.5, 1)),
                transforms.ColorJitter(contrast=(0, 1)),
                transforms.ColorJitter(saturation=(0, 1)),
                transforms.Grayscale(3),
                transforms.RandomResizedCrop(32, scale=(0.3, 1.0)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((-30, 30)),
                transforms.RandomAffine((-30, 30)),
                transforms.RandomAffine(0, translate=(0.1, 0.3)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]), 
            transforms.ToTensor()])
    elif _name == "mnist":
         transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]), 
            transforms.ToTensor()])
 
    elif _name == "test":
        transform = transforms.ToTensor()
    else:
        raise NameError("{} not found in transform loader".format(name))
    return transform


def load_checkpoint(model_dir, epoch=None, eval_=False):
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
    # net = torch.nn.DataParallel(net)
    del state_dict
    if eval_:
        net.eval()
    return net, epoch

    
def get_features(net, trainloader, verbose=True):
    '''extract all features out into one single batch. '''
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
    

def get_plabels(net, data, n_clusters=10, gamma=100):
    net.eval()
    transform = load_transforms('test')
    trainset = load_trainset(data, transform)
    trainloader = DataLoader(trainset, batch_size=500, num_workers=4)
    features, labels = get_features(net, trainloader)
    clustermd = ElasticNetSubspaceClustering(n_clusters=n_clusters, algorithm='spams', 
                                                gamma=gamma)
    clustermd.fit(features)
    plabels = clustermd.labels_
    accuracy = clustering_accuracy(labels, plabels)
    net.train()
    return plabels, accuracy


def corrupt_labels(trainset, ratio, seed):
    assert 0 <= ratio < 1, 'ratio should be between 0 and 1'
    num_classes = len(trainset.classes)
    num_corrupt = int(len(trainset.targets) * ratio)
    np.random.seed(seed)
    labels = trainset.targets
    indices = np.random.choice(len(labels), size=num_corrupt, replace=False)
    labels_ = np.copy(labels)
    for idx in indices:
        labels_[idx] = np.random.choice(np.delete(np.arange(num_classes), labels[idx]))
    trainset.targets = labels_
    return trainset


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Args:
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
    """Turn a membership matrix into a list of labels. """
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i, i])
    return labels

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
