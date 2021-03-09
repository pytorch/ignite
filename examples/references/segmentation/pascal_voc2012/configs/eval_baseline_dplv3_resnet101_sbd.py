# Basic training configuration
import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2 as ToTensor
from dataflow import get_inference_dataloader, ignore_mask_boundaries
from torchvision.models.segmentation import deeplabv3_resnet101

# ##############################
# Global configs
# ##############################

seed = 21
device = "cuda"
debug = False
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
    mode="test",
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


# baseline_dplv3_resnet101_sbd: best_model_78_val_miou_bg=0.6871.pt
weights_path = "d8b4687d86cf445a944853fdd6a6b999"
# or can specify a path
# weights_path = "/path/to/best_model.pt"
