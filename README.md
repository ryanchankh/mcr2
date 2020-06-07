# Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction
This repository is the official implementation of [Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction]()

## Requirements
- To install necessary python packages, run `pip install -r requirements.txt`
- For clustering, package `spam` is also required. See [here](https://pypi.org/project/spam/) for more details.

## Training
#### Supervised Setting
#### Unsupervised Setting
#### Contrastive Setting

## Evaluation
Methods available are: `svm`, `knn`, `nearsub`, `kmeans`, `ensc`. Each method also has options for testing hyperparameters, such as `--k` for top `k` components in kNN. Methods can also be chained. Checkpoint can also be specified using `--epoch` option.

An example for evaluation is:
```
python3 evaluate.py --knn --nearsub --k 10 --model_dir saved_models/sup_resnet18+128_cifar10_epo500_bs1000_lr0.001_mom0.9_wd0.0005_gam11.0_gam210.0_eps0.5_lcr0
```
, which runs kNN with top 10 components and nearest subspace on the latest checkpoint in `model_dir`

Refer to [code](./evaluate.py) for more implementation details. 


## Pretrain Models


