import argparse
import os
from tqdm import tqdm


import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC


import utils
import train_func as tf


def svm(args, train_features, train_labels, test_features, test_labels):
    print("Fitting LinearSVM model with {} samples".format(train_features.shape[0]))
    svm = LinearSVC(verbose=0)
    svm.fit(train_features, train_labels)
    acc = svm.score(test_features, test_labels)
    print("==> Test Accuracy - SVM: {}".format(acc))


def knn(args, train_features, train_labels, test_features, test_labels):
    ## Compute top k-nearest neighbors using cosine similarity
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred, test_labels)
    print("==> Test Accuracy - kNN: {}".format(acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')

    parser.add_argument('--k', type=int, default=5, help='top k components for kNN')
    args = parser.parse_args()

    ## load model, train data, and test data
    params = utils.load_params(args.model_dir)
    net, epoch = tf.load_checkpoint(args.model_dir, args.epoch)
    net = net.cuda().eval()
    
    train_transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True)
    if 'lcr' in params.keys(): # supervised corruption case
        trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
    trainloader = DataLoader(trainset, batch_size=500, shuffle=False, num_workers=4)
    train_features, train_labels = tf.get_features(net, trainloader)

    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)
    test_features, test_labels = tf.get_features(net, testloader)

    if args.svm:
        svm(args, train_features, train_labels, test_features, test_labels)
    if args.knn:
        knn(args, train_features, train_labels, test_features, test_labels)