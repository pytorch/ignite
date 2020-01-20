from typing import Type, Callable

import numpy as np

import cv2

from PIL import Image


from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCSegmentation
from torchvision.datasets.sbd import SBDataset


class TransformedDataset(Dataset):

    def __init__(self, ds: Dataset, transform_fn: Callable):
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

    def __init__(self, *args, return_meta: bool = False, **kwargs):
        super(VOCSegmentationOpencv, self).__init__(*args, **kwargs)
        self.return_meta = return_meta

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        assert img is not None, "Image at '{}' has a problem".format(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.asarray(Image.open(self.masks[index]))

        if self.return_meta:
            return {"image": img, "mask": mask,
                    "meta": {"index": index,
                             "image_path": self.images[index],
                             "mask_path": self.masks[index]
                             }
                    }

        return {"image": img, "mask": mask}


class SBDatasetOpencv(SBDataset):

    def __init__(self, *args, return_meta: bool = False, **kwargs):
        super(SBDatasetOpencv, self).__init__(*args, **kwargs)
        assert self.mode == "segmentation", "SBDatasetOpencv should be in segmentation mode only"
        self.return_meta = return_meta

    def _get_segmentation_target(self, filepath):
        mat = self._loadmat(filepath)
        return mat['GTcls'][0]['Segmentation'][0]

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        assert img is not None, "Image at '{}' has a problem".format(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self._get_target(self.masks[index])

        if self.return_meta:
            return {"image": img, "mask": mask,
                    "meta": {"index": index,
                             "image_path": self.images[index],
                             "mask_path": self.masks[index]
                             }
                    }

        return {"image": img, "mask": mask}


def get_train_dataset(root_path: str, return_meta: bool = False):
    return VOCSegmentationOpencv(root=root_path, year='2012', image_set='train', download=False,
                                 return_meta=return_meta)


def get_val_dataset(root_path: str, return_meta: bool = False):
    return VOCSegmentationOpencv(root=root_path, year='2012', image_set='val', download=False,
                                 return_meta=return_meta)


def get_train_noval_sbdataset(root_path: str, return_meta: bool = False):
    return SBDatasetOpencv(root_path, image_set='train_noval', mode='segmentation', return_meta=return_meta)
