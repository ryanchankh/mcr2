# Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction
This repository is the official implementation of [Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction](link).

## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.

## Training
### Basics
- All functions used in training can be found in [`train_func.py`](./train_func.py), which includes: `load_checkpoint(...)`, `load_trainset(...)`, etc. For implementation details please refer to docstring. 
- Code for training are in the following files: [`train_sup.py`](./train_sup.py) and [`train_selfsup.py`](./train_selfsup.py). Each has its own command options. 
- Augmentations is used in unsupervised and contrastive setting. Check [`augmentloader.py`](./augmentloader.py) for implementation details. 
- Our deep network architectures references [this repo](https://github.com/akamaster/pytorch_resnet_cifar10).

### Supervised Setting
#### Training Options
- Supervised Setting
```
usage: train_sup.py [-h] [--arch ARCH] [--fd FD] [--data DATA] [--epo EPO]
                    [--bs BS] [--lr LR] [--mom MOM] [--wd WD] [--gam1 GAM1]
                    [--gam2 GAM2] [--eps EPS] [--lcr LCR] [--lcs LCS]
                    [--tail TAIL] [--transform TRANSFORM]
                    [--save_dir SAVE_DIR] [--data_dir DATA_DIR]
                    [--pretrain_dir PRETRAIN_DIR]
                    [--pretrain_epo PRETRAIN_EPO]

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           architecture for deep neural network (default: resnet18)
  --fd FD               dimension of feature dimension (default: 128)
  --data DATA           dataset for training (default: CIFAR10)
  --epo EPO             number of epochs for training (default: 500)
  --bs BS               input batch size for training (default: 1000)
  --lr LR               learning rate (default: 0.0001)
  --mom MOM             momentum (default: 0.9)
  --wd WD               weight decay (default: 5e-4)
  --gam1 GAM1           gamma1 for tuning empirical loss (default: 1.)
  --gam2 GAM2           gamma2 for tuning empirical loss (default: 1.)
  --eps EPS             eps squared (default: 0.5)
  --lcr LCR             label corruption ratio (default: 0)
  --lcs LCS             label corruption seed for index randomization (default: 10)
  --tail TAIL           extra information to add to folder name
  --transform TRANSFORM transform applied to trainset (default: default
  --save_dir SAVE_DIR   base directory for saving PyTorch model. (default: ./saved_models/)
  --data_dir DATA_DIR   base directory for saving PyTorch model. (default: ./data/)
  --pretrain_dir PRETRAIN_DIR load pretrained checkpoint for assigning labels
  --pretrain_epo PRETRAIN_EPO load pretrained epoch for assigning labels
```

#### Examples
```
$ python3 train_sup.py --arch resnet18 --data cifar10 --fd 128 --epo 500 --bs 1000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.0001 --save_dir ./saved_models/
$ python3 train_sup.py --arch resnet18 --data cifar10 --fd 128 --epo 500 --bs 1000 --eps 1 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch vgg11 --data cifar10 --fd 128 --epo 500 --bs 1000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnext29_2x64d --data cifar10 --fd 128 --epo 500 --bs 1000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnet34 --data cifar10 --fd 128 --epo 500 --bs 1000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnet18 --data cifar10 --fd 128 --epo 500 --bs 500 --eps 0.5 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnet18 --data cifar10 --fd 128 --epo 500 --bs 4000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnet18stlsmall2 --data stl10 --fd 128 --epo 500 --bs 1000 --eps 0.5 --gam1 1 --gam2 1 --lr 0.05
$ python3 train_sup.py --arch resnet18stlsmall2 --data stl10 --fd 128 --epo 500 --bs 1000 --eps 1 --gam1 1 --gam2 1 --lr 0.01
$ python3 train_sup.py --arch resnet18stlsmall2 --data stl10_sup --fd 128 --epo 500 --bs 1000 --eps 1 --gam1 1 --gam2 1 --lr 0.001
```
### Self-supervised Setting
#### Training Options
```
usage: train_selfsup.py [-h] [--arch ARCH] [--fd FD] [--data DATA] [--epo EPO]
                        [--bs BS] [--aug AUG] [--lr LR] [--mom MOM] [--wd WD]
                        [--gam1 GAM1] [--gam2 GAM2] [--eps EPS] [--tail TAIL]
                        [--transform TRANSFORM] [--sampler SAMPLER]
                        [--pretrain_dir PRETRAIN_DIR]
                        [--pretrain_epo PRETRAIN_EPO] [--save_dir SAVE_DIR]
                        [--data_dir DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           architecture for deep neural network (default: resnet18)
  --fd FD               dimension of feature dimension (default: 32)
  --data DATA           dataset for training (default: CIFAR10)
  --epo EPO             number of epochs for training (default: 50)
  --bs BS               input batch size for training (default: 1000)
  --aug AUG             number of augmentations per mini-batch (default: 49)
  --lr LR               learning rate (default: 0.001)
  --mom MOM             momentum (default: 0.9)
  --wd WD               weight decay (default: 5e-4)
  --gam1 GAM1           gamma1 for tuning empirical loss (default: 1.0)
  --gam2 GAM2           gamma2 for tuning empirical loss (default: 10)
  --eps EPS             eps squared (default: 2)
  --tail TAIL           extra information to add to folder name
  --transform TRANSFORM transform applied to trainset (default: default
  --sampler SAMPLER     sampler used in augmentloader (default: random
  --pretrain_dir PRETRAIN_DIR load pretrained checkpoint for assigning labels
  --pretrain_epo PRETRAIN_EPO load pretrained epoch for assigning labels
  --save_dir SAVE_DIR   base directory for saving PyTorch model. (default: ./saved_models/)
  --data_dir DATA_DIR   base directory for saving PyTorch model. (default: ./data/)
```

#### Examples
```
$ python3 train_unsup.py --arch resnet18 --data cifar10 --fd 32 --epo 50 --bs 1000 --eps 2 --gam1 1 --gam2 0.5 --lr 0.02 --aug 50 --transform simclr
$ python3 train_unsup.py --arch resnet18 --data cifar10 --fd 32 --epo 50 --bs 1000 --eps 2 --gam1 1 --gam2 1 --lr 0.2 --aug 50 --transform simclr
$ python3 train_unsup.py --arch resnet18emp --data cifar10 --fd 128 --epo 150 --bs 1000 --eps 0.5 --gam1 15 --gam2 0.05 --lr 0.2 --aug 50 --transform simclr
$ python3 train_unsup.py --arch resnet18emp --data cifar10 --fd 128 --epo 150 --bs 1000 --eps 0.1 --gam1 15 --gam2 0.05 --lr 0.1 --aug 50 --transform simclr
$ python3 train_unsup.py --arch resnet18emp --data cifar10 --fd 128 --epo 150 --bs 1000 --eps 0.1 --gam1 15 --gam2 0.05 --lr 0.05 --aug 50 --transform simclr
$ python3 train_unsup.py --arch resnet18stlsmall --data stl10 --fd 128 --epo 150 --bs 1000 --eps 0.5 --gam1 20 --gam2 0.05 --lr 0.2 --aug 50 --transform stl10
$ python3 train_unsup.py --arch resnet18stlsmall --data stl10 --fd 128 --epo 150 --bs 1000 --eps 0.5 --gam1 15 --gam2 0.05 --lr 0.2 --aug 50 --transform stl10 
$ python3 train_unsup.py --arch resnet18stlsmall --data stl10 --fd 128 --epo 150 --bs 1000 --eps 0.5 --gam1 25 --gam2 0.05 --lr 0.2 --aug 50 --transform stl10 
$ python3 train_selfsup.py --arch resnet18stlsmall --data stl10 --fd 128 --epo 50 --bs 1000 --aug 50 --transform stl10 --sampler random  --eps 0.5 --gam1 20 --gam2 1 --lr 0.01
```


## Evaluation
Testing methods available are: `svm`, `knn`, `nearsub`, `kmeans`, `ensc`. Each method also has options for testing hyperparameters, such as `--k` for top `k` components in kNN. Methods can also be chained. Checkpoint can also be specified using `--epoch` option. Please refer to [`evaluate.py`](./evaluate.py) and [`./cluster.py`](./cluster.py) and for more implementation details. 

- Command Options
```
usage: evaluate.py [-h] [--model_dir MODEL_DIR] [--svm] [--knn] [--nearsub]
                   [--kmeans] [--ensc] [--epoch EPOCH] [--k K] [--n N]
                   [--gam GAM] [--tau TAU] [--n_comp N_COMP] [--save]
                   [--data_dir DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR base directory for saving PyTorch model.
  --svm                 evaluate using SVM
  --knn                 evaluate using kNN measuring cosine similarity
  --nearsub             evaluate using Nearest Subspace
  --kmeans              evaluate using KMeans
  --ensc                evaluate using Elastic Net Subspace Clustering
  --epoch EPOCH         which epoch for evaluation
  --k K                 top k components for kNN
  --n N                 number of clusters for cluster (default: 10)
  --gam GAM             gamma paramter for subspace clustering (default: 100)
  --tau TAU             tau paramter for subspace clustering (default: 1.0)
  --n_comp N_COMP       number of components for PCA (default: 30)
  --save                save labels
  --data_dir DATA_DIR   path to dataset
```

- An example for evaluation:
```
$ python3 evaluate.py --knn --nearsub --k 10 --model_dir saved_models/sup_resnet18+128_cifar10_epo500_bs1000_lr0.001_mom0.9_wd0.0005_gam11.0_gam210.0_eps0.5_lcr0
```
, which runs kNN with top 10 components and nearest subspace on the latest checkpoint in `model_dir`.


## Pretrain Models



## Lisence and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github. 


