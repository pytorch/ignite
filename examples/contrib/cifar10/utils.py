import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import models
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

import fastresnet


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_train_test_loaders(path, batch_size, num_workers, distributed=False, pin_memory=True):

    train_transform = Compose([
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    train_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_labelled_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_labelled_loader, test_loader


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in fastresnet.__dict__:
        fn = fastresnet.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn()
