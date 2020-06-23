from typing import Callable, Optional, Tuple, Union

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, ConcatDataset

import ignite.distributed as idist

from dataflow.datasets import get_train_dataset, get_val_dataset, TransformedDataset, get_train_noval_sbdataset


def get_train_val_loaders(
    root_path: str,
    train_transforms: Callable,
    val_transforms: Callable,
    batch_size: int = 16,
    num_workers: int = 8,
    val_batch_size: Optional[int] = None,
    with_sbd: Optional[str] = None,
    limit_train_num_samples: Optional[int] = None,
    limit_val_num_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = get_train_dataset(root_path)
    val_ds = get_val_dataset(root_path)

    if with_sbd is not None:
        sbd_train_ds = get_train_noval_sbdataset(with_sbd)
        train_ds = ConcatDataset([train_ds, sbd_train_ds])

    if limit_train_num_samples is not None:
        np.random.seed(limit_train_num_samples)
        train_indices = np.random.permutation(len(train_ds))[:limit_train_num_samples]
        train_ds = Subset(train_ds, train_indices)

    if limit_val_num_samples is not None:
        np.random.seed(limit_val_num_samples)
        val_indices = np.random.permutation(len(val_ds))[:limit_val_num_samples]
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        np.random.seed(len(val_ds))
        train_eval_indices = np.random.permutation(len(train_ds))[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    train_loader = idist.auto_dataloader(
        train_ds, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,
    )

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = idist.auto_dataloader(
        val_ds, shuffle=False, batch_size=val_batch_size, num_workers=num_workers, drop_last=False,
    )

    train_eval_loader = idist.auto_dataloader(
        train_eval_ds, shuffle=False, batch_size=val_batch_size, num_workers=num_workers, drop_last=False,
    )

    return train_loader, val_loader, train_eval_loader


def get_inference_dataloader(
    root_path: str,
    mode: str,
    transforms: Callable,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
    limit_num_samples: Optional[int] = None,
) -> DataLoader:
    assert mode in ("train", "test"), "Mode should be 'train' or 'test'"

    get_dataset_fn = get_train_dataset if mode == "train" else get_val_dataset

    dataset = get_dataset_fn(root_path, return_meta=True)

    if limit_num_samples is not None:
        indices = np.random.permutation(len(dataset))[:limit_num_samples]
        dataset = Subset(dataset, indices)

    dataset = TransformedDataset(dataset, transform_fn=transforms)

    loader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return loader
