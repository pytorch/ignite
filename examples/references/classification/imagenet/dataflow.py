from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder

import ignite.distributed as idist
from ignite.utils import convert_tensor


def opencv_loader(path):
    img = cv2.imread(path)
    assert img is not None, f"Image at '{path}' has a problem"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_dataloader(dataset, sampler=None, shuffle=False, limit_num_samples=None, **kwargs):
    if limit_num_samples is not None:
        g = torch.Generator().manual_seed(limit_num_samples)
        indices = torch.randperm(len(dataset), generator=g)[:limit_num_samples]
        dataset = Subset(dataset, indices)

    return idist.auto_dataloader(dataset, sampler=sampler, shuffle=(sampler is None) and shuffle, **kwargs)


def get_train_val_loaders(
    root_path: str,
    train_transforms: Callable,
    val_transforms: Callable,
    batch_size: int = 16,
    num_workers: int = 8,
    val_batch_size: Optional[int] = None,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ImageFolder(
        Path(root_path) / "train",
        transform=lambda sample: train_transforms(image=sample)["image"],
        loader=opencv_loader,
    )
    val_ds = ImageFolder(
        Path(root_path) / "val", transform=lambda sample: val_transforms(image=sample)["image"], loader=opencv_loader
    )

    if len(val_ds) < len(train_ds):
        g = torch.Generator().manual_seed(len(train_ds))
        train_eval_indices = torch.randperm(len(train_ds), generator=g)[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size

    train_loader = get_dataloader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        limit_num_samples=limit_train_num_samples,
    )

    val_loader = get_dataloader(
        val_ds,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=num_workers,
        drop_last=False,
        limit_num_samples=limit_val_num_samples,
    )

    train_eval_loader = get_dataloader(
        train_eval_ds,
        shuffle=False,
        batch_size=val_batch_size,
        num_workers=num_workers,
        drop_last=False,
        limit_num_samples=limit_val_num_samples,
    )

    return train_loader, val_loader, train_eval_loader


def denormalize(t, mean, std, max_pixel_value=255):
    assert isinstance(t, torch.Tensor), f"{type(t)}"
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor


def prepare_batch(batch, device, non_blocking):
    x, y = batch[0], batch[1]
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking)
    return x, y
