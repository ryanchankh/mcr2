import argparse
import os
import torch
import numpy as np

import train_func as tf
import utils

from shutil import copyfile


old_base = '/home/dxl/ryanchankh/compressible_learning/saved_models/'
model_dir = "unsup_res18Mod128_gam115.0_gam21.0_eps0.5_batch20_epoch400_aug_lr0.01_wd0.0005_numaug49_simCLR"
new_base = './saved_models/'
old_dir = os.path.join(old_base, model_dir)
new_dir = os.path.join(new_base, "old+"+model_dir)
ckpts = [paths for paths in os.listdir(old_dir) if paths[-3:] == ".pt"]


utils.init_pipeline(new_dir)

from architectures.resnet_cifar import *


params = utils.load_params(old_dir)
params["arch"] = 'resnet18old'
params['data'] = 'cifar10'
utils.save_params(new_dir, params)

for epoch in range(len(ckpts)):
    ckpt_path = os.path.join(old_dir, "model-epoch{}.pt".format(epoch))
    net = ResNet18Mod(128) 
    state_dict = torch.load(ckpt_path)
    net.load_state_dict(state_dict)
    net = torch.nn.DataParallel(net).cuda()
    utils.save_ckpt(new_dir, net, epoch)
    print(epoch)

print(new_dir)