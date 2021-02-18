# Basic training configuration
import os
from pathlib import Path
from functools import partial

import albumentations as A
import cv2
import torch.nn as nn
from albumentations.pytorch import ToTensorV2 as ToTensor
from dataflow import get_inference_dataloader
from dataflow import ignore_mask_boundaries
from torchvision.models.segmentation import deeplabv3_resnet101

# ##############################
# Global configs
# ##############################

seed = 21
device = "cuda"
debug = True
# Use AMP with torch native
with_amp = True


num_classes = 21
batch_size = 9  # total batch size
num_workers = 8  # total num workers per node

val_img_size = 513

# ##############################
# Setup Dataflow
# ##############################

assert "DATASET_PATH" in os.environ
data_path = os.environ["DATASET_PATH"]

assert "SBD_DATASET_PATH" in os.environ
sbd_data_path = os.environ["SBD_DATASET_PATH"]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


val_transforms = A.Compose(
    [
        A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=mean, std=std),
        ignore_mask_boundaries,
        ToTensor(),
    ]
)


data_loader = get_inference_dataloader(
    root_path=data_path,
    mode='test',
    transforms=val_transforms,
    batch_size=batch_size,
    num_workers=num_workers,
    limit_num_samples=batch_size * 5 if debug else None,
)

# ##############################
# Setup model
# ##############################

num_classes = 21
model = deeplabv3_resnet101(num_classes=num_classes)


def model_output_transform(output):
    return output["out"]


# training_task_id = "33ebd1b3c85a4e74b4450df646e224a7"
# weights_path = "baseline_dplv3_resnet101_sbd: best_model_60_val_miou_bg=0.6726.pt"
weights_path = "/mnt/data2/2f8e1ee50ee57b47eeb5ca3f97d26edd.best_model_0.pt"
