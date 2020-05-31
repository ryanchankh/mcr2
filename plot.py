import argparse
import os
import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader

from generate import gen_testloss, gen_accuracy
from loss import CompressibleLoss
import train_func as tf
import utils


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
    # com_loss_o = data['compress_loss_ortho'].ravel()
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
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.set_title("Theoretical Loss")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    file_name = os.path.join(loss_dir, 'loss_theoretical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_theoretical.pdf')
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
    # if len(com_loss_o) != 0:
        # ax.plot(num_iter, com_loss_o, label=r'ortho', 
            # color='black', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Empirical Loss")
    plt.tight_layout()
    file_name = os.path.join(loss_dir, 'loss_empirical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_empirical.pdf')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_loss_paper(args):
    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_

    ## extract loss from csv
    file_dir = os.path.join(args.model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)
    dis_loss_t = data['discrimn_loss_t'].ravel()
    com_loss_t = data['compress_loss_t'].ravel()
    obj_loss_t = dis_loss_t - com_loss_t

    avg_dis_loss_t, min_dis_loss_t, max_dis_loss_t = moving_average(dis_loss_t)
    avg_com_loss_t, min_com_loss_t, max_com_loss_t = moving_average(com_loss_t)
    avg_obj_loss_t, min_obj_loss_t, max_obj_loss_t = moving_average(obj_loss_t)

    ## Theoretical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(1, len(obj_loss_t))
    ax.plot(np.log(num_iter), avg_obj_loss_t[:-1], label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(np.log(num_iter), avg_dis_loss_t[:-1], label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(np.log(num_iter), avg_com_loss_t[:-1], label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    # ax.fill_between(np.log(num_iter), max_obj_loss_t[:-1], min_obj_loss_t[:-1], facecolor='green', alpha=0.5)
    # ax.fill_between(np.log(num_iter), max_dis_loss_t[:-1], min_dis_loss_t[:-1], facecolor='royalblue', alpha=0.5)
    # ax.fill_between(np.log(num_iter), max_com_loss_t[:-1], min_com_loss_t[:-1], facecolor='coral', alpha=0.5)
    ax.vlines(4, ymin=0, ymax=80, linestyle="--", linewidth=1.0, color='gray', alpha=0.8)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Number of iterations ($\log_2$ scale)', fontsize=14)
    ax.legend(loc='lower right', prop={"size": 14}, ncol=3, framealpha=0.5)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    plt.tight_layout()

    loss_dir = os.path.join(args.model_dir, "loss_paper")
    file_name = os.path.join(loss_dir, 'loss_theoretical.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss_theoretical.pdf')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_pca(args, features, labels, epoch):
    ## create save folder
    pca_dir = os.path.join(args.model_dir, 'figures', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)

    ## perform PCA on features
    n_comp = np.min([args.comp, features.shape[1]])
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=len(trainset.classes), stack=False)
    pca = PCA(n_components=n_comp).fit(features.numpy())
    sig_vals = [pca.singular_values_]
    for c in range(len(trainset.classes)): 
        pca = PCA(n_components=n_comp).fit(features_sort[c])
        sig_vals.append((pca.singular_values_))

    ## plot features
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=500)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    ax.plot(np.arange(x_min), sig_vals[0][:x_min], '-p', markersize=3, markeredgecolor='black',
        linewidth=1.5, color='tomato')
    map_vir = plt.cm.get_cmap('Blues', 6)
    norm = plt.Normalize(-10, 10)
    class_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    norm_class = norm(class_list)
    color = map_vir(norm_class)
    for c, sig_val in enumerate(sig_vals[1:]):
        ax.plot(np.arange(x_min), sig_val[:x_min], '-o', markersize=3, markeredgecolor='black',
                alpha=0.6, linewidth=1.0, color=color[c])
    ax.set_xticks(np.arange(0, x_min, 5))
    ax.set_yticks(np.arange(0, 35, 5))
    # for c, sig_val in enumerate(sig_vals[1:]):
    #     ax.plot(np.arange(x_min), sig_val[:x_min], marker='o', markersize=2, 
    #                 label=f'class - {c}', alpha=0.6)
    # ax.legend(loc='upper right', frameon=True, fancybox=True, prop={"size": 12}, ncol=2, framealpha=0.5)
    ax.set_xlabel("components", fontsize=14)
    ax.set_ylabel("sigular values", fontsize=14)
    # ax.set_title(f"PCA on features (Epoch: {epoch})")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    # ax.grid(True, color='white')
    # ax.set_facecolor('whitesmoke')
    fig.tight_layout()
    
    np.save(os.path.join(pca_dir, "sig_vals.npy"), sig_vals)
    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(pca_dir, f"pca_classVclass_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_hist(args, features, labels, epoch):
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)

    num_classes = labels.numpy().max() + 1
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


def plot_hist_all(args, features, labels, epoch):
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist_all')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    if not os.path.exists(os.path.join(hist_folder, "sim_mat")):
        os.makedirs(os.path.join(hist_folder, "sim_mat"))
    

    num_classes = len(trainset.classes)
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] 
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(10, 10), dpi=250, sharex=True, sharey=True)
    for j in range(10):
        for i in range(10):
            sim_mat = features_sort[i] @ features_sort[j].T
            if i == j:
                sim_mat = sim_mat[np.triu_indices(sim_mat.shape[0], k=1)]
                h1 = ax[i, j].hist(sim_mat, bins=np.arange(10)-0.05, color='green', alpha=0.3, histtype='bar', density=True)
            else:
                sim_mat = sim_mat.reshape(-1)
                h2 = ax[i, j].hist(sim_mat, bins=np.arange(10)-0.05, color='red', alpha=0.3, histtype='bar', density=True)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False) 
            
            ax[i, j].grid(True, color='white', axis='y')
            ax[i, j].set_xlim(0, 1.)
            ax[i, j].set_facecolor('whitesmoke')
            
            if i == 0:
                ax[i, j].set_title(f'class {j}')
            if j == 0:
                ax[i, j].set_ylabel(f'class {i}')
            
            np.save(os.path.join(hist_folder, "sim_mat", f"sim_mat{i}_{j}.npy"), sim_mat)
    fig.text(0.5, -0.01, 'cosine similarity', ha='center')
    fig.text(-0.01, 0.5, 'count', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()

    file_name = os.path.join(hist_folder, f"hist_all_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(hist_folder, f"hist_all_epoch{epoch}.pdf")
    fig.savefig(file_name)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_hist_paper(args, features, labels, epoch):
    ## create save folder
    hist_folder = os.path.join(args.model_dir, 'figures', 'hist_paper')
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)

    num_classes = len(trainset.classes)
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    i = 0
    ## inside class
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=250)
    sim_mat = features_sort[i] @ features_sort[i].T
    sim_mat = sim_mat[np.triu_indices(sim_mat.shape[0], k=1)][-500:]
    ax.hist(sim_mat, bins=30, color='green', alpha=0.4, density=True)
    ax.grid(True, color='white', axis='y')
    ax.set_xlim(0, 1.)
    ax.set_facecolor('whitesmoke')
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    fig.tight_layout()

    file_name = os.path.join(hist_folder, f"hist_0v0.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(hist_folder, f"hist_0v0.pdf")
    fig.savefig(file_name)
    plt.close()
    print("Plot saved to: {}".format(file_name))

    ## outside class
    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=250)
    temp = []
    for j in range(num_classes):
        sim_mat = (features_sort[i] @ features_sort[j].T).reshape(-1)
        temp.append(sim_mat)
    temp = np.hstack(temp)
    ax.hist(temp, bins=30, color='red', alpha=0.4, density=True)
    ax.grid(True, color='white', axis='y')
    ax.set_xlim(0, 1.)
    ax.set_facecolor('whitesmoke')
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False) 
    fig.tight_layout()

    file_name = os.path.join(hist_folder, f"hist_0vall.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(hist_folder, f"hist_0vall.pdf")
    fig.savefig(file_name)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_traintest(args, path_test):
    def process_df(data):
        epochs = data['epoch'].ravel().max()
        mean_, max_, min_ = [], [], []
        for epoch in np.arange(epochs+1):
            row = data[data['epoch'] == epoch].drop(columns=['step', 'discrimn_loss_e', 'compress_loss_e'])
            mean_.append(row.mean())
            max_.append(row.max())
            min_.append(row.min())
        return pd.DataFrame(mean_), pd.DataFrame(max_), pd.DataFrame(min_)

    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_

    path_train = os.path.join(args.model_dir, 'losses.csv')
    path_test = os.path.join(args.model_dir, 'losses_test.csv')
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    df_train_mean, df_train_max, df_train_min = process_df(df_train)
    df_test_mean, df_test_max, df_test_min = process_df(df_test)

    train_dis_loss_mean = df_train_mean['discrimn_loss_t'].ravel()
    train_com_loss_mean = df_train_mean['compress_loss_t'].ravel()
    train_obj_loss_mean = train_dis_loss_mean - train_com_loss_mean
    train_dis_loss_max = df_train_max['discrimn_loss_t'].ravel()
    train_com_loss_max = df_train_max['compress_loss_t'].ravel()
    train_obj_loss_max = train_dis_loss_max - train_com_loss_max
    train_dis_loss_min = df_train_min['discrimn_loss_t'].ravel()
    train_com_loss_min = df_train_min['compress_loss_t'].ravel()
    train_obj_loss_min = train_dis_loss_min - train_com_loss_min

    test_dis_loss_mean = df_test_mean['discrimn_loss_t'].ravel()
    test_com_loss_mean = df_test_mean['compress_loss_t'].ravel()
    test_obj_loss_mean = test_dis_loss_mean - test_com_loss_mean
    test_dis_loss_max = df_test_max['discrimn_loss_t'].ravel()
    test_com_loss_max = df_test_max['compress_loss_t'].ravel()
    test_obj_loss_max = test_dis_loss_max - test_com_loss_max
    test_dis_loss_min = df_test_min['discrimn_loss_t'].ravel()
    test_com_loss_min = df_test_min['compress_loss_t'].ravel()
    test_obj_loss_min = test_dis_loss_min - test_com_loss_min

    train_obj_loss_mean = moving_average(train_obj_loss_mean)[0] 
    test_obj_loss_mean = moving_average(test_obj_loss_mean)[0]
    train_dis_loss_mean = moving_average(train_dis_loss_mean)[0]
    test_dis_loss_mean = moving_average(test_dis_loss_mean)[0]
    train_com_loss_mean = moving_average(train_com_loss_mean)[0]
    test_com_loss_mean = moving_average(test_com_loss_mean)[0]
    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(train_obj_loss_mean))
    ax.plot(num_iter, train_obj_loss_mean, label=r'$\Delta R$ (train)', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_obj_loss_mean, label='$\Delta R$ (test)', 
                color='green', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.plot(num_iter, train_dis_loss_mean, label='$R$ (train)', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_dis_loss_mean, label='$R$ (test)', 
                color='royalblue', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.plot(num_iter, train_com_loss_mean, label='$R^c$ (train)', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, test_com_loss_mean, label='$R^c$ (test)', 
                color='coral', linewidth=1.0, alpha=0.8, linestyle='--')
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fancybox=True, prop={"size": 12}, ncol=3, framealpha=0.5)
    ax.set_ylim(0, 80)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    save_dir = os.path.join(args.model_dir, 'figures', "traintest")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"loss_traintest.png")
    fig.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"loss_traintest.pdf")
    fig.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    plt.close()
    

def plot_pca_epoch(args):
    EPOCHS = [0, 10, 100, 500]

    params = utils.load_params(args.model_dir)
    transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], transforms)
    trainloader = DataLoader(trainset, batch_size=200, num_workers=4)

    sig_vals = []
    for epoch in EPOCHS:
        epoch_ = epoch - 1
        if epoch_ == -1: # randomly initialized
            net = tf.load_architectures(params['arch'], params['fd'])
        else:
            net, epoch = tf.load_checkpoint(args.model_dir, epoch=epoch_, eval_=True)
        features, labels = tf.get_features(net, trainloader)
        if args.class_ is not None:
            features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=len(trainset.classes), stack=False)
            features_ = features_sort[args.class_]
        else:
            features_ = features.numpy()
        n_comp = np.min([args.comp, features.shape[1]])
        pca = PCA(n_components=n_comp).fit(features_)
        sig_vals.append(pca.singular_values_)

    ## plot singular values
    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=400)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    if args.class_ is not None:
        ax.set_xticks(np.arange(0, x_min, 10))
        ax.set_yticks(np.linspace(0, 40, 9))
        ax.set_ylim(0, 40)
    else:
        ax.set_xticks(np.arange(0, x_min, 10))
        ax.set_yticks(np.linspace(0, 80, 9))
        ax.set_ylim(0, 90)
    for epoch, sig_val in zip(EPOCHS, sig_vals):
        ax.plot(np.arange(x_min), sig_val[:x_min], marker='', markersize=5, 
                    label=f'epoch - {epoch}', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fancybox=True, prop={"size": 8}, ncol=1, framealpha=0.5)
    ax.set_xlabel("components")
    ax.set_ylabel("sigular values")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    ax.grid(True, color='white')
    ax.set_facecolor('whitesmoke')
    fig.tight_layout()

    ## save
    save_dir = os.path.join(args.model_dir, 'figures', 'pca')
    np.save(os.path.join(save_dir, "sig_vals_epoch.npy"), sig_vals)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"pca_class{args.class_}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"pca_class{args.class_}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_accuracy(args, path):
    def moving_average(arr, size=(9, 9)):
        assert len(size) == 2
        mean_ = []
        min_ = []
        max_ = [] 
        for i in range(len(arr)):
            l, r = i-size[0], i+size[1]
            l, r = np.max([l, 0]), r + 1 #adjust bounds
            mean_.append(np.mean(arr[l:r]))
            min_.append(np.amin(arr[l:r]))
            max_.append(np.amax(arr[l:r]))
        return mean_, min_, max_
    df = pd.read_csv(path)
    acc_train = df['acc_train'].ravel()
    acc_test = df['acc_test'].ravel()
    epochs = np.arange(len(df))

    acc_train, _, _ = moving_average(acc_train)
    acc_test, _, _ = moving_average(acc_test)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=400)
    ax.plot(epochs, acc_train, label='train', alpha=0.6, color='lightcoral')
    ax.plot(epochs, acc_test, label='test', alpha=0.6, color='cornflowerblue')
    ax.legend(loc='lower right', frameon=True, fancybox=True, prop={"size": 14}, ncol=2, framealpha=0.5)
    ax.set_xlabel("epochs", fontsize=14)
    ax.set_ylabel("accuracy", fontsize=14)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    ## save
    save_dir = os.path.join(args.model_dir, 'figures', 'acc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"acc_traintest.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"acc_traintest.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_heatmap(args, features, labels, epoch):
    num_classes = len(trainset.classes)
    features_sort, _ = utils.sort_dataset(features.numpy(), labels.numpy(), 
                            num_classes=num_classes, stack=False)
    features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)

    plt.rc('text', usetex=False)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, 50000, 6))
    ax.set_yticks(np.linspace(0, 50000, 6))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    
    save_dir = os.path.join(args.model_dir, 'figures', 'heatmaps')
    # os.mkdir(save_dir+"/sim_mat/")
    for i in range(500):
        np.save(save_dir+f"/sim_mat/sim_mat{i}.npy", sim_mat[i*100:(i+1)*100])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(save_dir, f"heatmat_epoch{epoch}.pdf")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ploting')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--loss', help='plot losses from training', action='store_true')
    parser.add_argument('--loss_paper', help='plot losses from training', action='store_true')
    parser.add_argument('--hist', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--hist_all', help='plot histogram of cosine similarity of features for all classes', action='store_true')
    parser.add_argument('--hist_paper', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--pca', help='plot PCA singular values of feautres', action='store_true')
    parser.add_argument('--pca_epoch', help='plot PCA singular for different epochs', action='store_true')
    parser.add_argument('--acc', help='plot accuracy over epochs', action='store_true')
    parser.add_argument('--traintest', help='plot train and test loss comparison plot', action='store_true')
    parser.add_argument('--heat', help='plot heatmap of cosine similarity between samples', action='store_true')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--n', type=int, default=1000, help='number of samples')
    parser.add_argument('--comp', type=int, default=30, help='number of components for PCA (default: 30)')
    parser.add_argument('--class_', type=int, default=None, help='which class for PCA (default: None)')
    args = parser.parse_args()
    
    if args.loss:
        plot_loss(args)
    if args.loss_paper:
        plot_loss_paper(args)
    if args.pca_epoch:
        plot_pca_epoch(args)

    if args.traintest:
        path = os.path.join(args.model_dir, 'losses_test.csv')
        if not os.path.exists(path):
            gen_test(args)
        plot_traintest(args, path)
    if args.acc:
        path = os.path.join(args.model_dir, 'accuracy.csv')
        if not os.path.exists(path):
            gen_accuracy(args)
        plot_accuracy(args, path)


    if args.pca or args.hist or args.hist_all or args.hist_paper or args.heat:
        ## load data and model
        params = utils.load_params(args.model_dir)
        net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
        transforms = tf.load_transforms('test')
        trainset = tf.load_trainset(params['data'], transforms)
        if 'lcr' in params.keys(): # supervised corruption case
            trainset = tf.corrupt_labels(trainset, params['lcr'], params['lcs'])
        trainloader = DataLoader(trainset, batch_size=200, num_workers=4)
        features, labels = tf.get_features(net, trainloader)

    if args.pca:
        plot_pca(args, features, labels, epoch)
    if args.hist:
        plot_hist(args, features, labels, epoch)
    if args.hist_all:
        plot_hist_all(args, features, labels, epoch)
    if args.hist_paper:
        plot_hist_paper(args, features, labels, epoch)
    if args.heat:
        plot_heatmap(args, features, labels, epoch)
