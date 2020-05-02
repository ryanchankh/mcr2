import pandas as pd
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt




def plot_loss(args):
    file_dir = os.path.join(args.model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)
    # epoch,step,loss,discrimn_loss_e,compress_loss_e,discrimn_loss_t,compress_loss_t

    dis_loss_e = data['discrimn_loss_e'].ravel()
    com_loss_e = data['compress_loss_e'].ravel()
    obj_loss_e = -data['loss'].ravel()
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
    file_name = os.path.join(args.model_dir, 'figures', 'loss_theoretical')
    plt.savefig(file_name + ".png", dpi=400)
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
    file_name = os.path.join(args.model_dir, 'figures', 'loss_empirical')
    plt.savefig(file_name + ".png", dpi=400)
    plt.close()

    print("Plot saved to: {}".format(file_name))

def plot_pca(args):
    None

def plot_hist(args):
    None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loss Plot')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--loss', help='plot losses from training', action='store_true')
    parser.add_argument('--hist', help='plot histogram of cosine similarity of features', action='store_true')
    parser.add_argument('--pca', help='plot PCA singular values of feautres', action='store_true')
    args = parser.parse_args()
    
    if args.loss:
        plot_loss(args)