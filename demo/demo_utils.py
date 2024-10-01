import torch
import torch.nn as nn
import numpy as np
import torchattacks
from utils.eval import base, adv
import matplotlib.pyplot as plt 
model_path_dict = {
    "wrn-eps-0":["wrn_28_4", "./demo-models/wrn-clean.pth.tar"],
    "wrn-eps-4":["wrn_28_4", "./demo-models/wrn-eps-4.pth.tar"],
    "wrn-eps-8":["wrn_28_4", "./demo-models/wrn-eps-8.pth.tar"],
    "resnet-eps-0":["resnet18", "./demo-models/resnet18-clean.pth.tar"],
    "resnet-eps-4":["resnet18", "./demo-models/resnet18-eps-4.pth.tar"],
    "resnet-eps-8":["resnet18", "./demo-models/resnet18-eps-8.pth.tar"],
}

class AttackArgs():
    def __init__(self, delta=8/255, num_steps=10):
        self.epsilon = delta # In the paper we call the budget delta
        self.num_steps = num_steps
        self.step_size = 0.0078 # Same in all experiments 0.0078
        self.clip_min = 0
        self.clip_max = 1.0
        self.const_init = True
        self.print_freq = 100

def plot_sharpnes_loss(sharpness,losses):
    mean_fams = np.mean(sharpness, axis=0)
    mean_losses = np.mean(losses, axis=0)
    fig, ax = plt.subplots(1,2, figsize = (12,4))
    ax[0].plot(mean_fams, color='red', label="adv")
    ax[1].plot(mean_losses, color='red', label="adv")

    ax[0].set_title("Sharpness on trajectory")
    ax[1].set_title("CE-Loss on trajectory")

    ax[0].set_xlabel("Iterations")
    ax[1].set_xlabel("Iterations")

    ax[0].set_ylabel("Sharpness")
    ax[1].set_ylabel("CE-loss")

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)

def attack_model(model, attack_test_loader, attack, args, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    top1, top5 = base(model, device, attack_test_loader, criterion, args, writer=None, epoch=0, return_pred=False)

    model.eval()
    # evaluate acc on adv examples 
    if attack in [ "cw-stronger", "cw", "cw-early"]:
        if attack == "cw":
            atk = torchattacks.CW_NO(model, c=10, kappa=20, steps=25, lr=0.01) # NO stands for no early stopping
        elif attack == "cw-stronger":
            atk = torchattacks.CW_NO(model, c=20, kappa=40, steps=50, lr=0.01)
        elif attack == "cw-early":
            atk = torchattacks.CW(model, c=10, kappa=20, steps=25, lr=0.01)
        raw_images = []
        targets = []
        adv_images = []
        path=[]
        for i,(ims, lab) in enumerate(attack_test_loader):
            raw_images.append(ims)
            targets.append(lab)
            _adv_images,tmppath = atk(ims, lab)
            adv_images.append(_adv_images)
            path.append(torch.permute(tmppath,(1,0,2,3,4)))
        raw_images = torch.concatenate(raw_images)
        targets = torch.concatenate(targets)
        adv_images = torch.concatenate(adv_images)
        path = torch.concatenate(path,axis=0).unsqueeze(2)
    elif attack in ["gnoise","unoise"]:
        adv_images = []
        raw_images = []
        targets = []
        path = []   
        noise_gen = lambda x: torch.normal(mean=torch.zeros_like(x))*0.25+0.5 if attack=="gnoise" else torch.rand(size=x.shape)
        for i,(ims, lab) in enumerate(attack_test_loader):
            tmppath = []
            gnoise = torch.clamp(noise_gen(ims),0.0,1.0)
            raw_images.append(ims)
            targets.append(lab)
            for alpha in np.linspace(0,1.0,10,endpoint=True):
                tmppath.append((1-alpha)*ims + alpha*(gnoise))
            path.append(torch.stack(tmppath))
        
        path = torch.permute(torch.concatenate(path, axis=1),(1,0,2,3,4)).unsqueeze(2)
        targets = torch.concatenate(targets)
    else: # This is the pgd version that we used in the paper
        criterion = nn.CrossEntropyLoss()
        top1_adv, top5_adv, adv_images, targets, succ, path = adv(
            model, device, attack_test_loader, criterion, args, 0, return_path=True
        )
        path = torch.permute(torch.concatenate(path,axis=1),(1,0,2,3,4))
        path = path.unsqueeze(2)

    return path, targets