# Basic training configuration
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from torchvision import transforms
from torchvision.models.resnet import resnet50

from dataflow.dataloaders import get_train_val_loaders
from dataflow.transforms import denormalize

# ##############################
# Global configs
# ##############################

seed = 19
device = 'cuda'
debug = False

fp16_opt_level = "O2"
val_interval = 2

train_crop_size = 224
val_crop_size = 320

batch_size = 128  # batch size per local rank
num_workers = 10   # num_workers per local rank


# ##############################
# Setup Dataflow
# ##############################

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


train_loader, val_loader, train_eval_loader = get_train_val_loaders(
    data_path,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    val_batch_size=batch_size * 4,
    pin_memory=True,
    train_sampler='distributed',
    val_sampler='distributed'
)

# Image denormalization function to plot predictions with images
img_denormalize = partial(denormalize, mean=mean, std=std)

# ##############################
# Setup Model
# ##############################

model = resnet50(pretrained=False)


# ##############################
# Setup Solver
# ##############################

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

