# Basic training configuration

# ##############################
# Global configs
# ##############################

seed = 17
device = 'cuda'
debug = False

fp16_opt_level = "O2"
val_interval = 2 if not debug else 100

train_crop_size = 224
val_crop_size = int(train_crop_size / 0.7)

batch_size = 320  # batch size per local rank
num_workers = 10   # num_workers per local rank


# ##############################
# Setup Dataflow
# ##############################

import os
import numpy as np
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import transforms, datasets

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(train_crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(val_crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
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
                          sampler=train_sampler)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size * 4,
                        num_workers=num_workers,
                        sampler=val_sampler,
                        pin_memory=True)

train_eval_loader = DataLoader(train_eval_dataset,
                               batch_size=batch_size * 4,
                               num_workers=num_workers,
                               pin_memory=True,
                               sampler=train_eval_sampler)

# ##############################
# Setup Model
# ##############################

from torchvision.models.resnet import resnet18

model = resnet18(pretrained=False)


# ##############################
# Setup Solver
# ##############################

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.distributed as dist

num_epochs = 5

criterion = nn.CrossEntropyLoss()

accumulation_steps = 4

lr = 0.001 * (batch_size * dist.get_world_size() / 256.0)
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)

le = len(train_loader)
lr_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=le * num_epochs, eta_min=0)
