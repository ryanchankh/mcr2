import argparse
import glob
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from evaluate import svm
from loss import MaximalCodingRateReduction
import train_func as tf
import utils


def gen_testloss(args):
    # load data and model
    params = utils.load_params(args.model_dir)
    ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
    ckpt_paths = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
    ckpt_paths = np.sort(ckpt_paths)
    
    # csv
    headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
        "discrimn_loss_t",  "compress_loss_t"]
    csv_path = utils.create_csv(args.model_dir, 'losses_test.csv', headers)
    print('writing to:', csv_path)

    # load data
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False)
    testloader = DataLoader(testset, batch_size=params['bs'], shuffle=False, num_workers=4)
    
    # save loss
    criterion = MaximalCodingRateReduction(gam1=params['gam1'], gam2=params['gam2'], eps=params['eps'])
    for epoch, ckpt_path in enumerate(ckpt_paths):
        net, epoch = tf.load_checkpoint(args.model_dir, epoch=epoch, eval_=True)
        for step, (batch_imgs, batch_lbls) in enumerate(testloader):
            features = net(batch_imgs.cuda())
            loss, loss_empi, loss_theo = criterion(features, batch_lbls, 
                                            num_classes=len(testset.num_classes))
            utils.save_state(args.model_dir, epoch, step, loss.item(), 
                *loss_empi, *loss_theo, filename='losses_test.csv')
    print("Finished generating test loss.")


def gen_training_accuracy(args):
    # load data and model
    params = utils.load_params(args.model_dir)
    ckpt_dir = os.path.join(args.model_dir, 'checkpoints')
    ckpt_paths = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
    ckpt_paths = np.sort(ckpt_paths)
    
    # csv
    headers = ["epoch", "acc_train", "acc_test"]
    csv_path = utils.create_csv(args.model_dir, 'accuracy.csv', headers)

    for epoch, ckpt_paths in enumerate(ckpt_paths):
        if epoch % 5 != 0:
            continue
        net, epoch = tf.load_checkpoint(args.model_dir, epoch=epoch, eval_=True)
        # load data
        train_transforms = tf.load_transforms('test')
        trainset = tf.load_trainset(params['data'], train_transforms, train=True)
        trainloader = DataLoader(trainset, batch_size=500, num_workers=4)
        train_features, train_labels = tf.get_features(net, trainloader, verbose=False)

        test_transforms = tf.load_transforms('test')
        testset = tf.load_trainset(params['data'], test_transforms, train=False)
        testloader = DataLoader(testset, batch_size=500, num_workers=4)
        test_features, test_labels = tf.get_features(net, testloader, verbose=False)

        acc_train, acc_test = svm(args, train_features, train_labels, test_features, test_labels)
        utils.save_state(args.model_dir, epoch, acc_train, acc_test, filename='accuracy.csv')
    print("Finished generating accuracy.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating files')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--test', help='create losses_test.csv', action='store_true')
    parser.add_argument('--train_acc', help='create accuracy.csv for training accuracy', action='store_true')

    parser.add_argument('--n_comp', type=int, default=30, help='Number of components for SVD.')
    args = parser.parse_args()

    if args.test:
        gen_testloss(args)
    if args.train_acc:
        gen_training_accuracy(args)
