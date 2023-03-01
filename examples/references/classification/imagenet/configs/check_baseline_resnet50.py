# Basic training configuration
import os
from functools import partial

import albumentations as A
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from albumentations.pytorch import ToTensorV2 as ToTensor
from dataflow import denormalize, get_train_val_loaders
from torchvision.models.resnet import resnet50

import ignite.distributed as idist

# ##############################
# Global configs
# ##############################

seed = 19
device = "cuda"
debug = True

# config to measure time passed to prepare batches and report measured time before the training
benchmark_dataflow = True
benchmark_dataflow_num_iters = 100

train_crop_size = 224
val_crop_size = 320

batch_size = 64 * idist.get_world_size()  # total batch size
num_workers = 8
val_interval = 2
start_by_validation = True


# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = A.Compose(
    [
        A.RandomResizedCrop(train_crop_size, train_crop_size, scale=(0.08, 1.0)),
        A.HorizontalFlip(),
        A.CoarseDropout(max_height=32, max_width=32),
        A.HueSaturationValue(),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

val_transforms = A.Compose(
    [
        # https://github.com/facebookresearch/FixRes/blob/b27575208a7c48a3a6e0fa9efb57baa4021d1305/imnet_resnet50_scratch/transforms.py#L76
        A.Resize(int((256 / 224) * val_crop_size), int((256 / 224) * val_crop_size)),
        A.CenterCrop(val_crop_size, val_crop_size),
        A.Normalize(mean=mean, std=std),
        ToTensor(),
    ]
)

train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size,
    limit_train_num_samples=batch_size * 6 if debug else None,
    limit_val_num_samples=batch_size * 6 if debug else None,
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

# ##############################
# Setup Model
# ##############################

model = resnet50(weights=None)


# ##############################
# Setup Solver
# ##############################

num_epochs = 2

criterion = nn.CrossEntropyLoss()

le = len(train_loader)

base_lr = 0.1 * (batch_size / 256.0)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lrs.MultiStepLR(optimizer, milestones=[30 * le, 60 * le, 90 * le, 100 * le], gamma=0.1)
