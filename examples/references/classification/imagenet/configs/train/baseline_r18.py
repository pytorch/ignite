# Basic training configuration
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

from torchvision.models.resnet import resnet18

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import denormalize

# ##############################
# Global configs
# ##############################

seed = 17
device = 'cuda'
debug = False

# config to measure time passed to prepare batches and report measured time before the training
benchmark_dataflow = True
benchmark_dataflow_num_iters = 100

fp16_opt_level = "O2"
val_interval = 2

# According to https://arxiv.org/pdf/1906.06423.pdf
# Train size: 224 -> Test size: 320 = max accuracy on ImageNet with ResNet-50
train_crop_size = 224
val_crop_size = 320

batch_size = 64  # batch size per local rank
num_workers = 8  # num_workers per local rank


# ##############################
# Setup Dataflow
# ##############################

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.RandomResizedCrop(train_crop_size, train_crop_size),
    A.HorizontalFlip(),
    A.CoarseDropout(max_height=32, max_width=32),
    A.HueSaturationValue(),
    A.Normalize(mean=mean, std=std),
    ToTensor(),
])

s = int((256 / train_crop_size) * val_crop_size)

val_transforms = A.Compose([
    # https://github.com/facebookresearch/FixRes/blob/b27575208a7c48a3a6e0fa9efb57baa4021d1305/imnet_resnet50_scratch/transforms.py#L76
    A.Resize(s, s),
    A.CenterCrop(val_crop_size, val_crop_size),
    A.Normalize(mean=mean, std=std),
    ToTensor(),
])

train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size,
    pin_memory=True,
    train_sampler='distributed',
    val_sampler='distributed'
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

# ##############################
# Setup Model
# ##############################

model = resnet18(pretrained=False)


# ##############################
# Setup Solver
# ##############################

num_epochs = 105

criterion = nn.CrossEntropyLoss()

le = len(train_loader)

base_lr = 0.1 * (batch_size * dist.get_world_size() / 256.0)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
lr_scheduler = lrs.MultiStepLR(optimizer, milestones=[30 * le, 60 * le, 90 * le, 100 * le], gamma=0.1)
