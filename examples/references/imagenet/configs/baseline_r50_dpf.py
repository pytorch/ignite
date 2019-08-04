# Basic training configuration

# ##############################
# Global configs
# ##############################

seed = 12
device = 'cuda'
debug = False

fp16_opt_level = "O2"
val_interval = 5 if not debug else 100

train_crop_size = 224
val_crop_size = int(train_crop_size / 0.7)

batch_size = 128  # batch size per local rank
num_workers = 4  # num_workers per local rank


# ##############################
# Setup Dataflow
# ##############################

import os
import numpy as np
from torch.utils.data import DataLoader, Subset, DistributedSampler

from torchvision import transforms, datasets
from dataflow import fast_collate, DataPrefetcher

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(train_crop_size),
    transforms.RandomHorizontalFlip(),
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(val_crop_size),
])


train_dataset = datasets.ImageNet(data_path, split='train', transform=train_transforms)
val_dataset = datasets.ImageNet(data_path, split='val', transform=val_transforms)

np.random.seed(seed)
train_eval_indices = np.random.permutation(len(train_dataset))[:len(val_dataset)]
train_eval_dataset = Subset(train_dataset, train_eval_indices)


train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_eval_sampler = DistributedSampler(train_eval_dataset)


train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=train_sampler,
                          collate_fn=fast_collate)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size * 4,
                        num_workers=num_workers,
                        sampler=val_sampler,
                        pin_memory=True,
                        collate_fn=fast_collate)

train_eval_loader = DataLoader(train_eval_dataset,
                               batch_size=batch_size * 4,
                               num_workers=num_workers,
                               pin_memory=True,
                               sampler=train_eval_sampler,
                               collate_fn=fast_collate)


mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

train_loader = DataPrefetcher(train_loader, mean=mean, std=std)
val_loader = DataPrefetcher(val_loader, mean=mean, std=std)
train_eval_loader = DataPrefetcher(train_eval_loader, mean=mean, std=std)


# ##############################
# Setup Model
# ##############################

from torchvision.models.resnet import resnet50

model = resnet50(pretrained=False)


# ##############################
# Setup Solver
# ##############################

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

num_epochs = 100

criterion = nn.CrossEntropyLoss()

accumulation_steps = 4

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)

le = len(train_loader)
milestones = [
    int(num_epochs * f * le) for f in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]
]
lr_scheduler = lrs.MultiStepLR(optimizer,
                               milestones=milestones,
                               gamma=0.5)

