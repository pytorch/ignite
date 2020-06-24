from typing import Mapping, Tuple

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader

import torchvision.models as M
import torchvision.transforms as T
from torchvision.datasets import FakeData

import ignite.distributed as idist


def initialize(config: Mapping) -> Tuple[nn.Module, optim.Optimizer, nn.Module, _LRScheduler]:
    # Adapt the code to your task

    model = getattr(M, config["model"])(num_classes=config["num_classes"])
    model = idist.auto_model(model)  # helper method to adapt model to current computation configuration

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    criterion = nn.CrossEntropyLoss()

    lr_scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    return model, optimizer, criterion, lr_scheduler


def get_dataflow(config: Mapping) -> Tuple[DataLoader, DataLoader]:
    # Adapt the code to your task

    num_classes = config["num_classes"]

    train_transform = T.Compose(
        [T.Pad(32), T.RandomResizedCrop(224), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]
    )
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    train_dataset = FakeData(size=500, num_classes=num_classes, transform=train_transform)
    val_dataset = FakeData(size=150, num_classes=num_classes, transform=val_transform)

    # helper method to create DataLoader adapted to current computation configuration
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=True
    )

    # helper method to create DataLoader adapted to current computation configuration
    val_loader = idist.auto_dataloader(
        val_dataset, batch_size=config["batch_size"] * 2, num_workers=config["num_workers"], shuffle=False
    )

    return train_loader, val_loader
