import argparse
import os

import numpy as np
from torch.optim import SGD

import train_func as tf
from augmentloader import AugmentLoader
from loss import CompressibleLoss2
import utils



parser = argparse.ArgumentParser(description='Self-supervised Learning')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18')
parser.add_argument('--fd', type=int, default=32,
                    help='dimension of feature dimension (default: 32)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=400,
                    help='number of epochs for training (default: 400)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--aug', type=int, default=49,
                    help='number of augmentations per mini-batch (default: 49)')
parser.add_argument('--nc', type=int, default=20,
                    help='number of clusters for pseudolabels (default: 20)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=1.0,
                    help='gamma1 for tuning empirical loss (default: 1.0)')
parser.add_argument('--gam2', type=float, default=10,
                    help='gamma2 for tuning empirical loss (default: 10)')
parser.add_argument('--gam3', type=float, default=10,
                    help='gamma3 for tuning empirical loss (default: 10)')
parser.add_argument('--eps', type=float, default=2,
                    help='eps squared (default: 2)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--sampler', type=str, default='balance',
                    help='sampler used in augmentloader (default: balance')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--savedir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
args = parser.parse_args()


## Pipelines Setup
model_dir = os.path.join(args.savedir,
               'selfsup_{}+{}_{}_epo{}_bs{}_aug{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_gam3{}_eps{}{}'.format(
                    args.arch, args.fd, args.data, args.epo, args.bs, args.aug, args.lr, 
                    args.mom, args.wd, args.gam1, args.gam2, args.gam3, args.eps, args.tail))
headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
            "discrimn_loss_t",  "compress_loss_t", "compress_loss_ortho"]
utils.init_pipeline(model_dir, headers=headers)
utils.save_params(model_dir, vars(args))


## per model functions
def lr_schedule(epoch, optimizer):
    """decrease the learning rate."""
    lr = args.lr
    if epoch > 10:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plabels_schedule(epoch, net, data, trainloader, n_clusters=10, gamma=100):
    """update labels using clutsering."""
    if epoch % 5 == 0: #test
        plabels, accuracy = tf.get_plabels(net, data, n_clusters=n_clusters, gamma=100)
        trainloader.update_labels(plabels)
        utils.save_labels(model_dir, plabels, epoch)
        print(f'epoch {epoch} - accuracy {accuracy}')


## Prepare for Training
if args.pretrain_dir is not None:
    net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)  
else:
    net = tf.load_architectures(args.arch, args.fd)
transforms = tf.load_transforms(args.transform)
trainset = tf.load_trainset(args.data)
trainloader = AugmentLoader(trainset,
                            transforms=transforms,
                            sampler=args.sampler,
                            batch_size=args.bs,
                            num_aug=args.aug)
criterion = CompressibleLoss2(gam1=args.gam1, gam2=args.gam2, gam3=args.gam3, eps=args.eps, num_aug=args.aug)
optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)


## Training
for epoch in range(args.epo):
    # lr_schedule(epoch, optimizer)
    # plabels_schedule(epoch, net, args.data, trainloader, n_clusters=args.nc)
    for step, (batch_imgs, batch_lbls, batch_idx) in enumerate(trainloader):
        batch_features = net(batch_imgs.cuda())
        loss, loss_empi, loss_theo, loss_ortho = criterion(batch_features, batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10:
            utils.save_ckpt(model_dir, net, epoch)
        utils.save_state(model_dir, epoch, step, loss.item(), *loss_empi, *loss_theo, loss_ortho)
    utils.save_ckpt(model_dir, net, epoch)
print("training complete.")
