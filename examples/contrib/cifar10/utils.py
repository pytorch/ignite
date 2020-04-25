import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.optim as optim

from torchvision import models
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

import fastresnet


device = "cuda"


def get_train_test_loaders(path, batch_size, num_workers, distributed=False, pin_memory=True):

    train_transform = Compose(
        [
            Pad(4),
            RandomCrop(32, fill=128),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

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

    train_labelled_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_labelled_loader, test_loader


def get_model(name):
    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in fastresnet.__dict__:
        fn = fastresnet.__dict__[name]
    else:
        raise RuntimeError("Unknown model name {}".format(name))

    return fn()


def get_model_optimizer(config, distributed=False):
    local_rank = config["local_rank"]
    if distributed:
        torch.cuda.set_device(local_rank)

    model = get_model(config["model"])
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank, ])
    elif torch.cuda.device_count() > 0:
        model = nn.parallel.DataParallel(model)

    return model, optimizer


def get_dataflow(config, distributed=False):

    # Rescale batch_size and num_workers
    ngpus_per_node = torch.cuda.device_count() if distributed else 1
    ngpus = dist.get_world_size() if distributed else 1
    batch_size = config["batch_size"] // ngpus
    num_workers = int((config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, test_loader = get_train_test_loaders(
        path=config["data_path"], batch_size=batch_size, distributed=distributed, num_workers=num_workers
    )
    return train_loader, test_loader


def get_default_config():
    batch_size = 512
    num_epochs = 24
    # Default configuration dictionary
    config = {
        "seed": 12,

        "data_path": "/tmp/cifar10",
        "output_path": "/tmp/output-cifar10",

        "model": "fastresnet",

        "batch_size": batch_size,
        "num_workers": 10,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "num_epochs": num_epochs,
        "learning_rate": 0.04,
        "num_warmup_epochs": 4,

        "validate_every": 3,

        # distributed settings
        "dist_url": "env://",
        "dist_backend": None,  # if None distributed option is disabled, set to "nccl" to enable

        # Logging:
        "display_iters": True,
        "log_model_grads_every": None,
        "checkpoint_every": 200,

        # Crash/Resume training:
        "resume_from": None,  # Path to checkpoint file .pt
        "crash_iteration": None,

        # Deterministic training: replace Engine by DeterministicEngine
        "deterministic": False
    }
    return config
