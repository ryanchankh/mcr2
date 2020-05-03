import argparse
import os
from tqdm import tqdm


import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC


import utils
import train_func as tf


def svm(args):
    params = utils.load_params(args.model_dir)
    net = tf.load_architectures(params['arch'], params['fd']).cuda()
    net, epoch = tf.load_checkpoint(args.model_dir, net, args.epoch)
    
    train_transforms = tf.load_transforms('default')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True)
    if 'lcr' in params.keys(): # supervised corruption case
        trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
    trainloader = DataLoader(trainset, batch_size=200, shuffle=True, num_workers=4)
    train_features, train_labels = tf.get_features(net, trainloader)

    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=200, shuffle=True, num_workers=4)
    test_features, test_labels = tf.get_features(net, testloader)

    print("Fitting LinearSVM model with {} samples".format(train_features.shape[0]))
    svm = LinearSVC(verbose=1)
    svm.fit(train_features, train_labels)
    acc = svm.score(test_features, test_labels)
    print("==> Test Accuracy - SVM: {}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    args = parser.parse_args()

    if args.svm:
        svm(args)