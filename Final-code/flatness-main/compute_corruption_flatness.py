# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import models
from data.cifar import *
from autoattack import AutoAttack
from args import parse_args
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.semisup import get_semisup_dataloader
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
    snip_init,
)
from utils.eval import hessian, adv
from utils.model import subnet_to_dense

# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning
import warnings

warnings.filterwarnings("ignore")
torch.set_warn_always(False)
def main():
    args = parse_args()
    print(args.val_method)
    parse_configs_file(args)
    print(args.val_method)

    # sanity checks
    
    assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    #result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    # add logger
    logger = logging.getLogger()
    logger.handlers = [] # This is the key thing for the question!

    # Start defining and assigning your handlers here
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create model
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
        ).to(device)
    #logger.info(model)

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)
    
    # Setup tensorboard writer
    #writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # Dataloader
    D = CIFAR10_C(args, normalize=args.normalize)
    test_loader = D.data_loaders()

    #logger.info( str(len(train_loader.dataset)), str(len(test_loader.dataset)))

    # Semi-sup dataloader
    if args.is_semisup:
        logger.info("Using semi-supervised training")
        sm_loader = get_semisup_dataloader(args, D.tr_train)
    else:
        sm_loader = None

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    #logger.info([criterion, optimizer, lr_policy])

    # train & val method

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            #logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            checkpoint["state_dict"] = subnet_to_dense(checkpoint["state_dict"],args.k)
            model.load_state_dict(checkpoint["state_dict"])
            #logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))




    if args.source_net:
        last_ckpt = checkpoint["state_dict"]
    else:
        last_ckpt = copy.deepcopy(model.state_dict())

    # Start testing
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate
        
        # evaluate flatness on clean examples
        model.eval() 
        top1, top5, fams_clean, norms_clean, hessians_clean = hessian(
            model, device, test_loader, criterion, args, epoch, args.fm
        )
        

        #sw, ss = sanity_check_paramter_updates(model, last_ckpt)
    logger.info("######## {} ########".format(args.name))
    logger.info(f"Clean: Top1  {top1}, Top5  {top5}, FAM-Mean {np.mean(fams_clean)}, FAM-Std {np.std(fams_clean)}, Hessian-Mean {np.mean(hessians_clean)}, Hessian-Std {np.std(hessians_clean)}, Norm-Mean {np.mean(norms_clean)}, Norm-Std {np.std(norms_clean)}")

    def compute_features(fams):
        features = []
        for i in range(10):
            features.append(np.sum(fams[:,i*256:(i+1)*256],axis=1).reshape(-1,1))
        features = np.concatenate(features,axis=1)
        #print(features.shape)
        return features
    
    if args.fm == "diagonal":
        weight_norm = norms_clean[0]
        fams_clean = np.concatenate(fams_clean)
        features_clean = weight_norm*compute_features(fams_clean)
        print(features_clean.shape)
        np.save("fams/corrupted/diagonal/features_{}_{}_{}.npy".format(args.name,args.corruption, args.severity), features_clean)
        np.save("fams/corrupted/diagonal/diag_{}_{}_{}.npy".format(args.name,args.corruption, args.severity), fams_clean)
        #print("Clean",features_clean)
        #print("Adv",features_adv)
    else:
        pass
    """
        prefix = "{}_{}_{}".format(args.attack,args.name,args.fm)
        mini = min(np.min(fams_clean),np.min(fams_adv))
        maxi = max(np.max(fams_clean),np.max(fams_adv))/2
        
        fig,ax = plt.subplots(1,1)
        ax.hist(fams_clean, bins=100, range=(mini,maxi), color='green', label="clean")
        ax.hist(fams_adv, bins=100, range=(mini,maxi), color='red', label="adv")
        ax.legend()
        fig.savefig("plots/{}.png".format(prefix), dpi=300)
        np.save("fams/{}_clean.npy".format(prefix), fams_clean)
        np.save("fams/{}_adv.npy".format(prefix), fams_adv)
        np.save("fams/{}_adv_succ.npy".format(prefix), succ.detach().cpu())

        mini = min(np.min(hessians_clean),np.min(hessians_adv))
        maxi = max(np.max(hessians_clean),np.max(hessians_adv))/2
        
        fig,ax = plt.subplots(1,1)
        ax.hist(hessians_clean, bins=100, range=(mini,maxi), color='green', label="clean")
        ax.hist(hessians_adv, bins=100, range=(mini,maxi), color='red', label="adv")
        ax.legend()
        fig.savefig("plots/hessian_{}.png".format(prefix), dpi=300)
        np.save("fams/hessian_{}_clean.npy".format(prefix), hessians_clean)
        np.save("fams/hessian_{}_adv.npy".format(prefix), hessians_adv)

        mini = min(np.min(norms_clean),np.min(norms_adv))
        maxi = max(np.max(norms_clean),np.max(norms_adv))
        
        fig,ax = plt.subplots(1,1)
        ax.hist(norms_clean, bins=100, range=(mini,maxi), color='green', label="clean")
        ax.hist(norms_adv, bins=100, range=(mini,maxi), color='red', label="adv")
        ax.legend()
        fig.savefig("plots/norms_{}.png".format(prefix), dpi=300)
        np.save("fams/norms_{}_clean.npy".format(prefix), norms_clean)
        np.save("fams/norms_{}_adv.npy".format(prefix), norms_adv)
"""
if __name__ == "__main__":
    main()
