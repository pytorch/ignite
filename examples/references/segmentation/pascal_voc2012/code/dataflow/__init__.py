# Baseline segmentation model on ISPRS datasets
from pathlib import Path
from functools import partial

import cv2
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

import albumentations as A

from polyaxon_client.tracking import get_outputs_refs_paths

from dataflow.transforms import prepare_batch_fp32, denormalize
from dataflow.datasets import get_isprs_train_val_test_datasets
from dataflow.dataloaders import get_train_val_loaders

from models import IsprsLWRefineNet

#################### Globals ####################

seed = 12
debug = False
device = 'cuda'
fp16_opt_level = "O2"

num_classes = 7

#################### Dataflow ####################

outputs_refs_paths = get_outputs_refs_paths()
assert outputs_refs_paths is not None
assert "jobs" in outputs_refs_paths and len(outputs_refs_paths['jobs']) == 3, "{}".format(outputs_refs_paths)

pdm_path = Path(outputs_refs_paths['jobs'][0]) / "potsdam_tiles"
vgn_path = Path(outputs_refs_paths['jobs'][1]) / "vaihingen_tiles"

pdm_csv = Path(outputs_refs_paths['jobs'][2]) / "potsdam" / "dataset_stats_with_fold_indices.csv"
vgn_csv = Path(outputs_refs_paths['jobs'][2]) / "vaihingen" / "dataset_stats_with_fold_indices.csv"

# According to https://arxiv.org/pdf/1906.06423.pdf
# Train size: 224 -> Test size: 320 = max accuracy on ImageNet with ResNet-50
val_img_size = 512
train_img_size = int(val_img_size * 0.72)

batch_size = 16
num_workers = 12 // dist.get_world_size()
val_batch_size = 16

mean = (0.5, 0.5, 0.5)
std = (0.25, 0.25, 0.25)

train_folds = [2, 3, 4, ]
val_folds = [1, ]
test_folds = [0, ]

train_ds, val_ds, test_ds = get_isprs_train_val_test_datasets(pdm_path, vgn_path,
                                                              pdm_csv, vgn_csv,
                                                              train_folds, val_folds, test_folds,
                                                              return_meta=False)

train_transforms = A.Compose([
    A.RandomResizedCrop(train_size, train_size),

    A.OneOf([
        A.Flip(),
        A.RandomRotate90(),
    ]),

    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2()
])

val_transforms = A.Compose([

    A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
    ToTensorV2()
])

train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    train_ds, val_ds,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=val_batch_size,
    pin_memory=True
)

#################### Model ####################

model = IsprsLWRefineNet(num_channels=3, num_classes=num_classes)

#################### Solver ####################

num_epochs = 100

criterion = nn.CrossEntropyLoss()

lr = 0.007
weight_decay = 5e-4
momentum = 0.9
nesterov = True
optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

le = len(train_loader)


def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)


lr_scheduler = lrs.LambdaLR(optimizer,
                            lr_lambda=partial(lambda_lr_scheduler, lr0=lr * 10.0, n=num_epochs * le, a=0.9))
