import argparse
import os
from tqdm import tqdm


import numpy as np
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import utils
import train_func as tf
from cluster import kmeans, ensc


def svm(args, train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(args, train_features, train_labels, test_features, test_labels):
    ## Compute top k-nearest neighbors using cosine similarity
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc


def nearsub(args, train_features, train_labels, test_features, test_labels):
    scores_pca = []
    scores_svd = []
    num_classes = train_labels.numpy().max() + 1
    features_sort, _ = utils.sort_dataset(train_features.numpy(), train_labels.numpy(), num_classes=num_classes, stack=False)
    for j in range(num_classes):
        pca = PCA(n_components=args.n_comp).fit(features_sort[j]) 
        subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        score_pca_j = np.linalg.norm((np.eye(params["fd"]) - subspace @ subspace.T) @ (test_features.numpy() - mean).T, ord=2, axis=0)

        pca = TruncatedSVD(n_components=args.n_comp).fit(features_sort[j])
        subspace = pca.components_.T
        score_svd_j = np.linalg.norm((np.eye(params["fd"]) - subspace @ subspace.T) @ (test_features.numpy()).T, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = utils.compute_accuracy(test_predict_pca, test_labels.numpy())
    acc_svd = utils.compute_accuracy(test_predict_svd, test_labels.numpy())
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_pca



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
    parser.add_argument('--kmeans', help='evaluate using KMeans', action='store_true')
    parser.add_argument('--ensc', help='evaluate using Elastic Net Subspace Clustering', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')

    parser.add_argument('--k', type=int, default=5, help='top k components for kNN')
    parser.add_argument('--n_comp', type=int, default=30, help='number of components for PCA (default: 30)')
    args = parser.parse_args()

    ## load model, train data, and test data
    params = utils.load_params(args.model_dir)
    net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
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
    if args.nearsub:
        nearsub(args, train_features, train_labels, test_features, test_labels)
    if args.kmeans:
        kmeans(args, train_features, train_labels)
    if args.ensc:
        ensc(args, train_features, train_labels)
    