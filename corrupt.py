import numpy as np
import torch

def default_corrupt(trainset, ratio, seed):
    """Corrupt labels in trainset.
    
    Parameters:
        trainset (torch.data.dataset): trainset where labels is stored
        ratio (float): ratio of labels to be corrupted. 0 to corrupt no labels; 
                            1 to corrupt all labels
        seed (int): random seed for reproducibility
        
    Returns:
        trainset (torch.data.dataset): trainset with updated corrupted labels
        
    """

    np.random.seed(seed)
    train_labels = np.asarray(trainset.targets)
    num_classes = np.max(train_labels) + 1
    n_train = len(train_labels)
    n_rand = int(len(trainset.data)*ratio)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    train_labels[randomize_indices] = np.random.choice(np.arange(num_classes), size=n_rand, replace=True)
    trainset.targets = torch.tensor(train_labels).int()
    return trainset


## https://github.com/shengliu66/ELR/blob/909687a4621b742cb5b8b44872d5bc6fce38bdd3/ELR/data_loader/cifar10.py#L82
def asymmetric_noise(trainset, ratio, seed):
    assert 0 <= ratio <= 1., 'ratio is bounded between 0 and 1' 
    np.random.seed(seed)
    train_labels = np.array(trainset.targets)
    train_labels_gt = train_labels.copy()
    for i in range(trainset.num_classes):
        indices = np.where(train_labels == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < ratio * len(indices):
#                 self.noise_indx.append(idx)
                # truck -> automobile
                if i == 9:
                    train_labels[idx] = 1
                # bird -> airplane
                elif i == 2:
                    train_labels[idx] = 0
                # cat -> dog
                elif i == 3:
                    train_labels[idx] = 5
                # dog -> cat
                elif i == 5:
                    train_labels[idx] = 3
                # deer -> horse
                elif i == 4:
                    train_labels[idx] = 7
    trainset.targets = torch.tensor(train_labels).int()
    return trainset
                    

# https://github.com/xiaoboxia/T-Revision/blob/b984283b884c13eb59ed0f8d435f4eda548ab26a/data/utils.py#L125
# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(trainset, noise, seed=None):
    """mistakes:
        flip in the pair
    """
    y_train = np.array(trainset.targets)
    nb_classes = np.unique(trainset.targets).size
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
#         print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    
    trainset.targets = torch.tensor(y_train)
    return trainset

# https://github.com/xiaoboxia/T-Revision/blob/b984283b884c13eb59ed0f8d435f4eda548ab26a/data/utils.py#L149
def noisify_multiclass_symmetric(trainset, noise, seed=10):
    """mistakes:
        flip in the symmetric way
    """
    y_train = np.array(trainset.targets)
    nb_classes = np.unique(y_train).size
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=seed)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
#         print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    
    trainset.targets = torch.tensor(y_train)
    return trainset



#### Helper 
def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#     print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert np.allclose(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
#     print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y
