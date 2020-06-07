# Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction
This repository is the official implementation of [Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction]()

## Requirements
- To install necessary python packages, run `pip install -r requirements.txt`
- For clustering, package `spam` is also required. See [here](https://pypi.org/project/spam/) for more details.

## Training
### Basics
- All functions used in training can be found in [`train_func.py`](./train_func.py), which includes: `load_checkpoint()`, `load_trainset()`, etc. For implementation details please refer to docstring. 
- Code for training are in the following files: `train_sup.py`, `train_unsup.py`, `train_contrast.py`. Each has its own command options. 

### Examples
- Supervised Setting
```
python3 train_sup.py --arch resnet18 --data cifar10 --fd 128 --epo 500 --bs 1000 --transform default --eps 0.5 --gam1 1 --gam2 1 --lr 0.0001 --save_dir ./saved_models/
```
- Unsupervised Setting
```
python3 train_unsup.py --arch resnet18stlsmall --data stl10 --fd 128 --epo 50 --bs 1000 --aug 50 --transform stl10 --sampler random  --eps 0.5 --gam1 20 --gam2 1 --lr 0.01
```
- Contrastive Setting
```
TBD
```

## Evaluation
Testing methods available are: `svm`, `knn`, `nearsub`, `kmeans`, `ensc`. Each method also has options for testing hyperparameters, such as `--k` for top `k` components in kNN. Methods can also be chained. Checkpoint can also be specified using `--epoch` option.

An example for evaluation is:
```
python3 evaluate.py --knn --nearsub --k 10 --model_dir saved_models/sup_resnet18+128_cifar10_epo500_bs1000_lr0.001_mom0.9_wd0.0005_gam11.0_gam210.0_eps0.5_lcr0
```
, which runs kNN with top 10 components and nearest subspace on the latest checkpoint in `model_dir`.

Refer to [code](./evaluate.py) for more implementation details. 


## Pretrain Models


