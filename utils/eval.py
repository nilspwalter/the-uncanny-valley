import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F

from utils.logging import AverageMeter, ProgressMeter
from utils.adv import pgd_whitebox, fgsm
from models.resnet_cifar import ResNet
from models.vgg_cifar import VGG
from scipy.stats import norm
import numpy as np
import time
from utils.FAMloss import *


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def hessian(model, device, val_loader, criterion, args,  epoch=0, fm="neuronwise"):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    fam = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    end = time.time()
    fams = []
    norms = []
    hessians = []
    for i, data in enumerate(val_loader):
            #if i > 2048:
            #    break
            images, target = data[0].to(device), data[1].to(device)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # wrn 103
            # resnet 60
            layer_hessian = LayerHessian(model, 103, F.cross_entropy, method="autograd_fast")
            weights_norm, hessian = FAMreg(images, target, layer_hessian, norm_function=fm, approximate=False)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #print(torch.mean((torch.argmax(output,dim=1)==target).float()))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            #print(len(regu.detach().cpu()))
            if fm=="layerwise_trace":
                regu = weights_norm * hessian
                fams.append(regu.detach().cpu())
                norms.append(weights_norm.detach().cpu())
                hessians.append(hessian.detach().cpu())
            elif fm=="diagonal":
                fams.append(hessian.detach().cpu().numpy().reshape(1,-1)) # TODO remove dirty hack
                norms.append(weights_norm.detach().cpu().numpy()) #legacy TODO remove this the norm is always the same
                hessians.append(hessian.detach().cpu().numpy())
            else:
                fams.append(regu.detach().cpu()[0])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)
    progress.display(i)  # print final results
    return top1.avg, top5.avg, fams, norms, hessians

def hessian_per_label(model, device, val_loader, criterion, args,  epoch=0, fm="layerwise_trace"):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    fam = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    # switch to evaluate mode
    end = time.time()
    fams = []
    norms = []
    hessians = []
    for i, data in enumerate(val_loader):
        images, target = data[0].to(device), data[1].to(device)

        # compute output
        #output = model(images)
        temp_fams = []
        temp_norms = []
        temp_hessians = []    
        for label in range(10):
            target_label = torch.zeros_like(target) + label
            #print(target_label,target)
            #loss = criterion(output, target)
            output = model(images)
            #print("CE",F.cross_entropy(output,target_label))
            layer_hessian = LayerHessian(model, 103, F.cross_entropy, method="autograd_fast")
            weights_norm, hessian = FAMreg(images, target_label, layer_hessian, norm_function=fm, approximate=False)
            regu = weights_norm * hessian
            temp_fams.append(regu.detach().cpu())
            #temp_norms.append(weights_norm.detach().cpu())
            temp_hessians.append(hessian.detach().cpu())
        fams.append(temp_fams)
        norms.append(temp_norms)
        hessians.append(temp_hessians)
        if (i + 1) % args.print_freq == 0:
            progress.display(i)
    progress.display(i)  # print final results
    return 0,0, np.array(fams), np.array(norms), np.array(hessians)


def hessian_per_path(model, device, val_loader, criterion, args,  epoch=0, fm="layerwise_trace", approximate=False):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    fam = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    end = time.time()
    fams = []
    norms = []
    hessians = []
    losses = []
    for i, data in enumerate(val_loader):
            trace_images, target = data[0][0].to(device), data[1].to(device)
            trace_fams = []
            trace_loss = []
            if i>2000:
                continue
            #print(20*"#")
            for j in range(trace_images.shape[0]):
                image = trace_images[j]
                output, phi = model(image, ret_act=True)
                probs = torch.softmax(output,dim=1)

                hessian = (probs*(1-probs)).sum() * (phi*phi).sum()
                if isinstance(model, torchvision.models.densenet.DenseNet):
                    weights_norm = torch.linalg.norm(model.classifier.weight)

                elif isinstance(model, ResNet):
                    weights_norm = torch.linalg.norm(model.linear.weight)
                elif isinstance(model, VGG):
                    weights_norm = torch.linalg.norm(model.classifier[4].weight)
                else:
                    weights_norm = torch.linalg.norm(model.fc.weight)
                loss = criterion(output, target)
                regu = weights_norm * hessian

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                trace_fams.append(regu.detach().cpu())
                trace_loss.append(loss.detach().cpu())

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            fams.append(trace_fams)
            losses.append(trace_loss)

            batch_time.update(time.time() - end)
            end = time.time()
            #print(20*"#")
            if (i + 1) % args.print_freq == 0:
                progress.display(i)
    progress.display(i)  # print final results

    return top1.avg, top5.avg, np.array(fams), np.array(losses)


def base(model, device, val_loader, criterion, args, writer, epoch=0, return_pred=False):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    preds = []
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            pred = torch.argmax(output,axis=1).cpu().numpy()
            preds.append(pred)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results
    if return_pred:
        return top1.avg, top5.avg, np.concatenate(preds)
    return top1.avg, top5.avg


def adv(model, device, val_loader, criterion, args, writer=None, epoch=0,return_path=False, ):
    """
        Evaluate on adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    adv_top5 = AverageMeter("Adv-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, adv_top1, adv_top5],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        all_images = []
        all_targets = []
        masks = []
        paths = []
        for i, data in enumerate(val_loader):
            #if i>500:
            #    continue
            images, target = data[0].to(device), data[1].to(device)
            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            correct_mask = (target==torch.argmax(output,axis=1)).to(torch.int)
            # adversarial images
            images, path = pgd_whitebox(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                args.clip_min,
                args.clip_max,
                is_random=not args.const_init,
                return_path=True
            )

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            adv_top1.update(acc1[0], images.size(0))
            adv_top5.update(acc5[0], images.size(0))
            adv_correct_mask = (target==torch.argmax(output,axis=1)).to(torch.int)

            succ_attack = (correct_mask - adv_correct_mask)==1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)
            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )
            all_images.append(images.clone())
            all_targets.append(target.clone())
            masks.append(succ_attack.clone())
            paths.append(torch.stack(path))
        progress.display(i)  # print final results
    mask = torch.cat(masks)

    if return_path:
        return adv_top1.avg, adv_top5.avg, torch.cat(all_images), torch.cat(all_targets), mask, paths
    else:
        return adv_top1.avg, adv_top5.avg, torch.cat(all_images), torch.cat(all_targets), mask

