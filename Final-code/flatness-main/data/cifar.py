import os
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class CIFAR10_C(datasets.CIFAR10):
    def __init__(self,args, normalize=False):
        self.args = args
        self.corruption = args.corruption
        assert self.corruption in ['pixelate', 'frost', 'contrast', 'brightness', 'fog', 'saturate','jpeg_compression',
                                     'elastic_transform','impulse_noise','zoom_blur','snow','speckle_noise','shot_noise',
                                     'gaussian_noise','glass_blur','gaussian_blur','motion_blur','defocus_blur','spatter']

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        #self.transform = transforms.Compose(self.tr_train)
        self.transform = transforms.Compose(self.tr_test)

        self.data = np.load(os.path.join(args.data_dir,  "CIFAR10-C",args.corruption + ".npy"))
        self.targets = np.load(os.path.join(args.data_dir, "CIFAR10-C", "labels.npy"))

        self.data = self.data[(args.severity-1)*10000:(args.severity)*10000 ]
        self.targets = self.targets[(args.severity-1)*10000:(args.severity)*10000 ]

    def data_loaders(self,**kwargs):
        self.target_transform = None
        self.test_loader = DataLoader(
            self, batch_size=self.args.test_batch_size, shuffle=False
        )

        return self.test_loader
    
class CIFAR10:
    """ 
        CIFAR-10 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]
        print(self.args.batch_size)
        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR10(
            root=os.path.join(self.args.data_dir, "CIFAR10"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader



class CIFAR100:
    """ 
        CIFAR-100 dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
        )

        self.tr_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
        self.tr_test = [transforms.ToTensor()]

        if normalize:
            print("Normalized")
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=True,
            download=True,
            transform=self.tr_train,
        )

        subset_indices = np.random.permutation(np.arange(len(trainset)))[
            : int(self.args.data_fraction * len(trainset))
        ]

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            **kwargs,
        )
        testset = datasets.CIFAR100(
            root=os.path.join(self.args.data_dir, "CIFAR100"),
            train=False,
            download=True,
            transform=self.tr_test,
        )
        test_loader = DataLoader(
            testset, batch_size=self.args.test_batch_size, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, test_loader