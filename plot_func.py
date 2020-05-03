import argparse
import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import utils
import train_func as tf


def plot_loss(args):
    ## create saving directory
    loss_dir = os.path.join(args.model_dir, 'figures', 'loss')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    file_dir = os.path.join(args.model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)

    ## extract loss from csv
    obj_loss_e = -data['loss'].ravel()
    dis_loss_e = data['discrimn_loss_e'].ravel()
    com_loss_e = data['compress_loss_e'].ravel()
    dis_loss_t = data['discrimn_loss_t'].ravel()
    com_loss_t = data['compress_loss_t'].ravel()
    obj_loss_t = dis_loss_t - com_loss_t

    ## Theoretical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(obj_loss_t))
    ax.plot(num_iter, obj_loss_t, label=r'$\mathcal{L}^d-\mathcal{L}^c$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, dis_loss_t, label=r'$\mathcal{L}^d$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, com_loss_t, label=r'$\mathcal{L}^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='upper left', prop={"size": 15}, framealpha=0.5)
    ax.set_title("Theoretical Loss")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    file_name = os.path.join(loss_dir, 'loss_theoretical.png')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))

    ## Empirial Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(obj_loss_e))
    ax.plot(num_iter, obj_loss_e, label=r'$\widehat{\mathcal{L}^d}-\widehat{\mathcal{L}^c}$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, dis_loss_e, label=r'$\widehat{\mathcal{L}^d}$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, com_loss_e, label=r'$\widehat{\mathcal{L}^c}$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='best', prop={"size": 15}, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Empirical Loss")
    plt.tight_layout()
    file_name = os.path.join(loss_dir, 'loss_empirical.png')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_pca(args, features, epoch):
    ## create save folder
    pca_dir = os.path.join(args.model_dir, 'figures', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    ## perform PCA on features
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=len(trainset.classes), stack=False)
    pca = PCA(n_components=args.comp).fit(features.numpy())
    sig_vals = [pca.singular_values_]
    for c in range(len(trainset.classes)): 
        pca = PCA(n_components=args.comp).fit(features_sort[c])
        sig_vals.append((pca.singular_values_))

    ## plot features
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=250)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    ax.plot(np.arange(x_min), sig_vals[0][:x_min], marker='x', markersize=5, color='black', 
                label=f'all', alpha=0.6)
    ax.set_xticks(np.arange(0, x_min, 2))
    ax.set_yticks(np.linspace(0, np.int32(np.max(sig_vals[0])), 10))
    for c, sig_val in enumerate(sig_vals[1:]):
        ax.plot(np.arange(x_min), sig_val[:x_min], marker='.', markersize=5, 
                    label=f'class {c}', alpha=0.6)
    ax.legend()
    ax.set_xlabel("components")
    ax.set_ylabel("sigular values")
    ax.set_title(f"PCA on features (Epoch: {epoch})")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    file_name = os.path.join(pca_dir, f"pca_epoch{epoch}.png")
    fig.savefig(file_name)
    plt.close()
    print("Plot saved to: {}".format(file_name))

def plot_hist(args, features_per_class, epoch):
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    else:
        files = glob.glob(hist_folder+"/*")
        for f in files:
            os.remove(f)

    num_classes = len(trainset.classes)
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    for i in range(num_classes):
        for j in range(i, num_classes):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=250)
            if i == j:
                sim_mat = features_sort[i] @ features_sort[j].T
                sim_mat = sim_mat[np.triu_indices(sim_mat.shape[0], k = 1)]
            else:
                sim_mat = (features_sort[i] @ features_sort[j].T).reshape(-1)
            ax.hist(sim_mat, bins=40, color='red', alpha=0.5)
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("count")
            ax.set_title(f"Class {i} vs. Class {j}")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig.tight_layout()

            file_name = os.path.join(hist_folder, f"hist_{i}v{j}")
            fig.savefig(file_name)
            plt.close()
            print("Plot saved to: {}".format(file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--loss', help='plot losses from training', action='store_true')
    parser.add_argument('--hist', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--pca', help='plot PCA singular values of feautres', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--comp', type=int, default=30, help='number of components for PCA (default: 30)')
    args = parser.parse_args()
    
    if args.loss:
        plot_loss(args)

    if args.pca or args.hist:
        ## load data and model
        params = utils.load_params(args.model_dir)
        net = tf.load_architectures(params['arch'], params['fd']).cuda()
        net, epoch = tf.load_checkpoint(args.model_dir, net, args.epoch)
        transforms = tf.load_transforms('test')
        trainset = tf.load_trainset(params['data'], transforms)
        if 'lcr' in params.keys(): # supervised corruption case
            trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
        trainloader = DataLoader(trainset, batch_size=200, shuffle=True, num_workers=4)
        features, labels = tf.get_features(net, trainloader)

    if args.pca:
        plot_pca(args, features, epoch)
    if args.hist:
        plot_hist(args, features, epoch)