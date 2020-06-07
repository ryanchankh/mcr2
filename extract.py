import argparse
import numpy as np
import time
import os
from tqdm import tqdm
import tarfile

from torch.utils.data import DataLoader
import utils
import train_func as tf  
    


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from model and data')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    parser.add_argument('--save_dir', type=str, default="./extractions/")
    parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to file name')

    args = parser.parse_args()
    params = utils.load_params(args.model_dir)
    net, epoch = tf.load_checkpoint(args.model_dir, args.epoch, eval_=True)
    train_transforms = tf.load_transforms('test')
    trainset = tf.load_trainset(params['data'], train_transforms, train=True)
    trainloader = DataLoader(trainset, batch_size=200, num_workers=4)
    features, labels = tf.get_features(net, trainloader)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, "features.npy"), features.cpu().detach().numpy())
    np.save(os.path.join(args.save_dir, "labels.npy"), labels.numpy())
    make_tarfile("./extractions.tgz", args.save_dir)