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
import data
from autoattack import AutoAttack
import torchattacks
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
from utils.eval import hessian, adv, hessian_per_path, base
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
    handler2 = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(handler2)

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
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()
    args.test_batch_size = 512
    D2 = data.__dict__[args.dataset](args, normalize=args.normalize)
    _, attack_test_loader = D2.data_loaders()
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
        model.eval()
        #top1, top5, fams_clean, norms_clean, hessians_clean = hessian(
        #    model, device, test_loader, criterion, args, epoch, args.fm
        #)
        top1, top5 = base(model, device, test_loader, criterion, args, writer=None, epoch=0, return_pred=False)
        model.eval()
        # evaluate acc on adv examples 
        if args.attack in ['apgd-ce' ,'apgd-dlr', 'fab' ,'square']:
            raw_images = []
            targets = []
            for i,(ims, lab) in enumerate(test_loader):
                raw_images.append(ims)
                targets.append(lab)
            raw_images = torch.concatenate(raw_images)
            targets = torch.concatenate(targets)

            adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=[args.attack])
            adv_images, adv_predict = adversary.run_standard_evaluation(raw_images, targets, bs=256, return_labels=True)
        elif args.attack in [ "cw-stronger","new-pgd", "apgd", "pifgsmpp", "spsa", "jitter", "vni", "cw"]:
            if args.attack == "new-pgd":
                atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
            elif args.attack == "apgd":
                atk = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
            elif args.attack == "pifgsmpp":
                atk = torchattacks.PIFGSMPP(model, max_epsilon=16/255, num_iter_set=10)
            elif args.attack == "spsa":
                atk = torchattacks.SPSA(model, eps=0.3)
            elif args.attack == "jitter":
                atk = torchattacks.Jitter(model, eps=8/255, alpha=2/255, steps=10, scale=10, std=0.1, random_start=True)
            elif args.attack == "vni":
                atk = torchattacks.VNIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, N=5, beta=3/2)
            elif args.attack == "cw":
                atk = torchattacks.CW(model, c=10, kappa=20, steps=100, lr=0.01)
            elif args.attack == "cw-stronger":
                atk = torchattacks.CW(model, c=20, kappa=40, steps=50, lr=0.01)
            raw_images = []
            targets = []
            adv_images = []
            path=[]
            print(len(attack_test_loader))
            for i,(ims, lab) in enumerate(attack_test_loader):
                raw_images.append(ims)
                targets.append(lab)
                _adv_images,tmppath = atk(ims, lab)
                adv_images.append(_adv_images)
                path.append(torch.permute(tmppath,(1,0,2,3,4)))
                print(i)
            raw_images = torch.concatenate(raw_images)
            targets = torch.concatenate(targets)
            adv_images = torch.concatenate(adv_images)
            path = torch.concatenate(path,axis=0).unsqueeze(2)
            print(path.shape)
        elif args.attack in ["gnoise","unoise"]:
            adv_images = []
            raw_images = []
            targets = []
            path = []
            for i,(ims, lab) in enumerate(attack_test_loader):
                if i > 1:
                    continue
                tmppath = []
                gnoise = torch.clamp((torch.normal(mean=torch.zeros_like(ims))*0.25+0.5),0.0,1.0)
                #print(gnoise.max(),gnoise.min(), gnoise.mean(), gnoise.std())
                #print(gnoise.shape)
                raw_images.append(ims)
                targets.append(lab)
                for alpha in np.linspace(0,1.0,10,endpoint=True):
                    tmppath.append((1-alpha)*ims + alpha*(gnoise))
                path.append(torch.stack(tmppath))
            
            path = torch.permute(torch.concatenate(path, axis=1),(1,0,2,3,4)).unsqueeze(2)
            targets = torch.concatenate(targets)
            print(path.shape)

        else:
            top1_adv, top5_adv, adv_images, targets, succ, path = adv(
                model, device, attack_test_loader, criterion, args, epoch, return_path=True
            )
            print(path[0].shape)
            path = torch.permute(torch.concatenate(path,axis=1),(1,0,2,3,4))
            path = path.unsqueeze(2)

        train = torch.utils.data.TensorDataset(path, targets)
        adv_loader = DataLoader(
            train, batch_size=1, shuffle=False)
        
        # evaluate acc on adv examples 
        top1_adv, top5_adv, adv_fams_trace, trace_losses = hessian_per_path(
            model, device, adv_loader, criterion, args, epoch, args.fm, args.approximate
        )  

        #sw, ss = sanity_check_paramter_updates(model, last_ckpt)
    logger.info("######## {} ########".format(args.name))
    logger.info(f"Adversarial: Top1  {top1_adv}, Top5  {top5_adv}, FAM-Mean {np.mean(adv_fams_trace[:,-1])}, FAM-Std {np.std(adv_fams_trace[:,-1])}")


    mini = 0
    maxi = 100
    print(adv_fams_trace.shape)
    print(adv_fams_trace.shape)
    mean_fams = np.mean(adv_fams_trace, axis=0)
    mean_losses = np.mean(trace_losses, axis=0)
    fig,ax = plt.subplots(1,2, figsize = (10,5))
    ax[0].plot(mean_fams, color='red', label="adv")
    ax[1].plot(mean_losses, color='red', label="adv")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("Norm * Hessian on PGD-trajectory" + " Adv-Acc-1 " + str(round(float(top1_adv),2))+ " Adv-Acc-5 " + str(round(float(top5_adv),2)))
    ax[1].set_title("CE-Loss on PGD-trajectory")
    fig.savefig("plots/trace/{}_{}_{}.png".format(args.attack,args.name,args.fm), dpi=300)
    np.save("fams/trace_paper/{}_{}_{}_adv.npy".format(args.attack,args.name,args.fm), adv_fams_trace)
    np.save("fams/trace_paper/{}_{}_{}_loss.npy".format(args.attack,args.name,args.fm), mean_losses)
    np.save("fams/trace_paper/{}_{}_{}_trace_loss.npy".format(args.attack,args.name,args.fm), trace_losses)
    np.save("fams/trace_paper/{}_{}_{}_accs.npy".format(args.attack,args.name,args.fm),{"CleanAcc-1":float(top1),"CleanAcc-5":float(top5),"AdvAcc-1":float(top1_adv),"AdvAcc-5":float(top5_adv)})
    #np.save("fams/trace/{}_{}_{}_adv_succ.npy".format(args.attack,args.name,args.fm), succ.detach().cpu())

    #fig,ax = plt.subplots(1,1)

    #ax.hist(hessians_adv, bins=100, range=(mini,maxi), color='red', label="adv")
    #ax.legend()
    #fig.savefig("plots/hessian_{}_{}.png".format(args.name,args.fm), dpi=300)

    #np.save("fams/hessian_{}_{}_adv.npy".format(args.name,args.fm), hessians_adv)

    
    #fig,ax = plt.subplots(1,1)
    #ax.legend()
    #fig.savefig("plots/norms_{}_{}.png".format(args.name,args.fm), dpi=300)
    #np.save("fams/norms_{}_{}_adv.npy".format(args.name,args.fm), norms_adv)

if __name__ == "__main__":
    main()
