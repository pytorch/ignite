from typing import Mapping

import torchvision.transforms as T
from torchvision.datasets import FakeData

import ignite.distributed as idist


def initialize(config: Mapping):
    pass


def get_dataflow(config: Mapping):

    train_transform = T.Compose(
        [T.Pad(32), T.RandomResizedCrop(224), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))]
    )
    val_transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    train_dataset = FakeData(size=1000, transform=train_transform)
    val_dataset = FakeData(size=300, transform=val_transform)

    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"],
    )

    pass
