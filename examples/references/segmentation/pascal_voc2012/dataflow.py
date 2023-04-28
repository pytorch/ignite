import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision.datasets.sbd import SBDataset
from torchvision.datasets.voc import VOCSegmentation

import ignite.distributed as idist
from ignite.utils import convert_tensor


class TransformedDataset(Dataset):
    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(**dp)


class VOCSegmentationOpencv(VOCSegmentation):
    target_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(self, *args, return_meta=False, **kwargs):
        super(VOCSegmentationOpencv, self).__init__(*args, **kwargs)
        self.return_meta = return_meta

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        assert img is not None, f"Image at '{self.images[index]}' has a problem"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(self.masks[index]))

        if self.return_meta:
            return {
                "image": img,
                "mask": mask,
                "meta": {"index": index, "image_path": self.images[index], "mask_path": self.masks[index]},
            }

        return {"image": img, "mask": mask}


class SBDatasetOpencv(SBDataset):
    def __init__(self, *args, return_meta=False, **kwargs):
        super(SBDatasetOpencv, self).__init__(*args, **kwargs)
        assert self.mode == "segmentation", "SBDatasetOpencv should be in segmentation mode only"
        self.return_meta = return_meta

    def _get_segmentation_target(self, filepath):
        mat = self._loadmat(filepath)
        return mat["GTcls"][0]["Segmentation"][0]

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        assert img is not None, f"Image at '{self.images[index]}' has a problem"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self._get_target(self.masks[index])

        if self.return_meta:
            return {
                "image": img,
                "mask": mask,
                "meta": {"index": index, "image_path": self.images[index], "mask_path": self.masks[index]},
            }

        return {"image": img, "mask": mask}


def get_train_dataset(root_path, return_meta=False):
    return VOCSegmentationOpencv(
        root=root_path, year="2012", image_set="train", download=False, return_meta=return_meta
    )


def get_val_dataset(root_path, return_meta=False):
    return VOCSegmentationOpencv(root=root_path, year="2012", image_set="val", download=False, return_meta=return_meta)


def get_train_noval_sbdataset(root_path, return_meta=False):
    return SBDatasetOpencv(root_path, image_set="train_noval", mode="segmentation", return_meta=return_meta)


def get_dataloader(dataset, sampler=None, shuffle=False, limit_num_samples=None, **kwargs):
    if limit_num_samples is not None:
        g = torch.Generator().manual_seed(limit_num_samples)
        indices = torch.randperm(len(dataset), generator=g)[:limit_num_samples]
        dataset = Subset(dataset, indices)

    return idist.auto_dataloader(dataset, sampler=sampler, shuffle=(sampler is None) and shuffle, **kwargs)


def get_train_val_loaders(
    root_path,
    train_transforms,
    val_transforms,
    batch_size=16,
    num_workers=8,
    train_sampler=None,
    val_batch_size=None,
    sbd_path=None,
    limit_train_num_samples=None,
    limit_val_num_samples=None,
):
    train_ds = get_train_dataset(root_path)
    val_ds = get_val_dataset(root_path)

    if sbd_path is not None:
        sbd_train_ds = get_train_noval_sbdataset(sbd_path)
        train_ds = train_ds + sbd_train_ds

    if len(val_ds) < len(train_ds):
        g = torch.Generator().manual_seed(len(train_ds))
        train_eval_indices = torch.randperm(len(train_ds), generator=g)[: len(val_ds)]
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size

    train_loader = get_dataloader(
        train_ds,
        shuffle=True,
        sampler=train_sampler,
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


def get_inference_dataloader(
    root_path, mode, transforms, batch_size=16, num_workers=8, pin_memory=True, limit_num_samples=None
):
    assert mode in ("train", "test"), "Mode should be 'train' or 'test'"

    get_dataset_fn = get_train_dataset if mode == "train" else get_val_dataset

    dataset = get_dataset_fn(root_path, return_meta=True)
    dataset = TransformedDataset(dataset, transform_fn=transforms)
    return get_dataloader(
        dataset,
        limit_num_samples=limit_num_samples,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def ignore_mask_boundaries(**kwargs):
    assert "mask" in kwargs, "Input should contain 'mask'"
    mask = kwargs["mask"]
    mask[mask == 255] = 0
    kwargs["mask"] = mask
    return kwargs


def denormalize(t, mean, std, max_pixel_value=255):
    assert isinstance(t, torch.Tensor), f"{type(t)}"
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    return tensor


def prepare_image_mask(batch, device, non_blocking):
    x, y = batch["image"], batch["mask"]
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y
