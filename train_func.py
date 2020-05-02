from tqdm import tqdm

import numpy as np
import torchvision
import torchvision.transforms as transforms

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
        from architectures.resnet_cifar import ResNet152
        net = ResNet18Mod(dim)
    else:
        raise NameError("{} not found in archiectures.".format(name))
    return net


def load_trainset(name, transform=None):
    _name = name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10/', train=True,
                                                download=True, transform=transform)
    elif _name == "cifar10":
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100/', train=True,
                                                 download=True, transform=transform)
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root="/Users/ryanchankh/Datasets/mnist/", 
                                              train=True, download=True, transform=transform)
    else:
        raise NameError("{} not found in trainset loader".format(name))
    return trainset


def load_transforms(name):
    _name = name.lower()
    if _name == "default":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif _name == "simclr":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    elif _name == "random":
         train_transform = transforms.Compose([
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
    else:
        raise NameError("{} not found in transform loader".format(name))
    return transforms


def load_dataloader(name, *args):
    _name = name.lower()
    print(args)

def get_features(net, trainloader):
    '''extract all features out into one single batch. '''
    features = []
    labels = []
    train_bar = tqdm(trainloader)
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        batch_features = net(batch_imgs.cuda())
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)


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