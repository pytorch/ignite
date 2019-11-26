from typing import Callable, Optional, Tuple, Union

import numpy as np
import cv2

from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Subset
import torch.utils.data.distributed as data_dist
from torchvision.datasets import ImageNet


def opencv_loader(path):
    img = cv2.imread(path)
    assert img is not None, "Image at '{}' has a problem".format(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_train_val_loaders(root_path: str,
                          train_transforms: Callable,
                          val_transforms: Callable,
                          batch_size: int = 16,
                          num_workers: int = 8,
                          val_batch_size: Optional[int] = None,
                          pin_memory: bool = True,
                          random_seed: Optional[int] = None,
                          train_sampler: Optional[Union[Sampler, str]] = None,
                          val_sampler: Optional[Union[Sampler, str]] = None,
                          limit_train_num_samples: Optional[int] = None,
                          limit_val_num_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = ImageNet(root_path, split='train',
                        transform=lambda sample: train_transforms(image=sample)['image'],
                        loader=opencv_loader)
    val_ds = ImageNet(root_path, split='val',
                      transform=lambda sample: val_transforms(image=sample)['image'],
                      loader=opencv_loader)

    if limit_train_num_samples is not None:
        if random_seed is not None:
            np.random.seed(random_seed)
        train_indices = np.random.permutation(len(train_ds))[:limit_train_num_samples]
        train_ds = Subset(train_ds, train_indices)

    if limit_val_num_samples is not None:
        val_indices = np.random.permutation(len(val_ds))[:limit_val_num_samples]
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) < len(train_ds):
        train_eval_indices = np.random.permutation(len(train_ds))[:len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    if isinstance(train_sampler, str):
        assert train_sampler == 'distributed'
        train_sampler = data_dist.DistributedSampler(train_ds)

    train_eval_sampler = None
    if isinstance(val_sampler, str):
        assert val_sampler == 'distributed'
        val_sampler = data_dist.DistributedSampler(val_ds, shuffle=False)
        train_eval_sampler = data_dist.DistributedSampler(train_eval_ds, shuffle=False)

    train_loader = DataLoader(train_ds, shuffle=train_sampler is None,
                              batch_size=batch_size, num_workers=num_workers,
                              sampler=train_sampler,
                              pin_memory=pin_memory, drop_last=True)

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler,
                            batch_size=val_batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False, sampler=train_eval_sampler,
                                   batch_size=val_batch_size, num_workers=num_workers,
                                   pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, train_eval_loader
