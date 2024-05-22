import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F

from utils.logging import AverageMeter, ProgressMeter
from utils.adv import pgd_whitebox, fgsm
from symbolic_interval.symbolic_network import (
    sym_interval_analyze,
    naive_interval_analyze,
    mix_interval_analyze,
)
from crown.bound_layers import (
    BoundSequential,
    BoundLinear,
    BoundConv2d,
    BoundDataParallel,
    Flatten,
)
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
            #if i>1000:
            #    continue
            #print(20*"#")
            for j in range(trace_images.shape[0]):
                #print(trace_images.shape)
                image = trace_images[j]
                # compute output
                output, phi = model(image, ret_act=True)
                probs = torch.softmax(output,dim=1)
                #print(probs)
                #print((probs*(1-probs)).sum())
                #print((phi*phi).sum())
                hessian = (probs*(1-probs)).sum() * (phi*phi).sum()
                #print(type(model))
                if isinstance(model, torchvision.models.densenet.DenseNet):
                    weights_norm = torch.linalg.norm(model.classifier.weight)

                elif isinstance(model, ResNet):
                    weights_norm = torch.linalg.norm(model.linear.weight)
                elif isinstance(model, VGG):
                    weights_norm = torch.linalg.norm(model.classifier[4].weight)
                else:
                    weights_norm = torch.linalg.norm(model.fc.weight)
                loss = criterion(output, target)
                #layer_hessian = LayerHessian(model, 103, F.cross_entropy, method="autograd_fast")
                #weights_norm, hessian = FAMreg(image, target, layer_hessian, norm_function=fm, approximate=approximate)
                regu = weights_norm * hessian
                #print("Fam", regu)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                trace_fams.append(regu.detach().cpu())
                trace_loss.append(loss.detach().cpu())
                #print(regu.shape)
                #norms.append(weights_norm.detach().cpu())
                #hessians.append(hessian.detach().cpu())
            #print(trace_fams)
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


def mixtrain(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    sym_losses = AverageMeter("Sym_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    sym_top1 = AverageMeter("Sym-Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, sym_losses, top1, top5, sym_top1],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            rce_avg = 0
            rerr_avg = 0
            for r in range(images.shape[0]):

                rce, rerr = sym_interval_analyze(
                    model,
                    args.epsilon,
                    images[r : r + 1],
                    target[r : r + 1],
                    use_cuda=torch.cuda.is_available(),
                    parallel=False,
                )
                rce_avg = rce_avg + rce.item()
                rerr_avg = rerr_avg + rerr

            rce_avg = rce_avg / float(images.shape[0])
            rerr_avg = rerr_avg / float(images.shape[0])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            sym_losses.update(rce_avg, images.size(0))
            sym_top1.update((1 - rerr_avg) * 100.0, images.size(0))

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
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return sym_top1.avg, sym_top1.avg


def ibp(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ibp_losses = AverageMeter("IBP_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    ibp_top1 = AverageMeter("IBP-Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, ibp_losses, top1, top5, ibp_top1],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            rce, rerr = naive_interval_analyze(
                model,
                args.epsilon,
                images,
                target,
                use_cuda=torch.cuda.is_available(),
                parallel=False,
            )

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            ibp_losses.update(rce.item(), images.size(0))
            ibp_top1.update((1 - rerr) * 100.0, images.size(0))

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
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  # print final results

    return ibp_top1.avg, ibp_top1.avg


def smooth(model, device, val_loader, criterion, args, writer, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    rad = AverageMeter("rad", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, top1, top5, rad], prefix="Smooth (eval): "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # Defult: evaluate on 10 random samples of additive gaussian noise.
            output = []
            for _ in range(10):
                # add noise
                if args.dataset == "imagenet":
                    std = (
                        torch.tensor([0.229, 0.224, 0.225])
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    ).to(device)
                    noise = (torch.randn_like(images) / std).to(device) * args.noise_std
                else:
                    noise = torch.randn_like(images).to(device) * args.noise_std

                output.append(F.softmax(model(images + noise), -1))

            output = torch.sum(torch.stack(output), axis=0)

            p_max, _ = output.max(dim=-1)
            radii = (args.noise_std + 1e-16) * norm.ppf(p_max.data.cpu().numpy())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            rad.update(np.mean(radii))

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
                    "Adv-test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )

        progress.display(i)  # print final results

    return top1.avg, rad.avg


def freeadv(model, device, val_loader, criterion, args, writer, epoch=0):

    assert (
        not args.normalize
    ), "Explicit normalization is done in the training loop, Dataset should have [0, 1] dynamic range."

    # Mean/Std for normalization
    mean = torch.Tensor(np.array(args.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, args.image_dim, args.image_dim).to(device)
    std = torch.Tensor(np.array(args.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, args.image_dim, args.image_dim).to(device)

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ",
    )

    eps = args.epsilon
    K = args.num_steps
    step = args.step_size
    model.eval()
    end = time.time()
    print(" PGD eps: {}, num-steps: {}, step-size: {} ".format(eps, K, step))
    for i, (input, target) in enumerate(val_loader):

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        orig_input = input.clone()
        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).to(device)
        input += randn
        input.clamp_(0, 1.0)
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input - eps, input)
            input = torch.min(orig_input + eps, input)
            input.clamp_(0, 1.0)

        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

        if writer:
            progress.write_to_tensorboard(writer, "test", epoch * len(val_loader) + i)

        # write a sample of test images to tensorboard (helpful for debugging)
        if i == 0 and writer:
            writer.add_image(
                "Adv-test-images",
                torchvision.utils.make_grid(input[0 : len(input) // 4]),
            )

    progress.display(i)  # print final results

    return top1.avg, top5.avg
