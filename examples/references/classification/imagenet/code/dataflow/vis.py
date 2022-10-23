from typing import Callable, Optional

import numpy as np
import torch

try:
    from image_dataset_viz import render_datapoint
except ImportError:
    raise ModuleNotFoundError(
        "Please install image-dataset-viz via pip install --upgrade git+https://github.com/vfdev-5/ImageDatasetViz.git"
    )


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().numpy().transpose((1, 2, 0))
    return img.astype(np.uint8)


def make_grid(
    batch_img: torch.Tensor,
    batch_preds: torch.Tensor,
    img_denormalize_fn: Callable,
    batch_gt: Optional[torch.Tensor] = None,
):
    """Create a grid from batch image and mask as

        i+l1+gt1  | i+l2+gt2  | i+l3+gt3  | i+l4+gt4  | ...

        where i+l+gt = image + predicted label + ground truth

    Args:
        batch_img (torch.Tensor) batch of images of any type
        batch_preds (torch.Tensor) batch of masks
        img_denormalize_fn (Callable): function to denormalize batch of images
        batch_gt (torch.Tensor, optional): batch of ground truth masks.
    """
    assert isinstance(batch_img, torch.Tensor) and isinstance(batch_preds, torch.Tensor)
    assert len(batch_img) == len(batch_preds), f"{len(batch_img)} vs {len(batch_preds)}"
    assert batch_preds.ndim == 1, f"{batch_preds.ndim}"

    if batch_gt is not None:
        assert isinstance(batch_gt, torch.Tensor)
        assert len(batch_preds) == len(batch_gt)
        assert batch_gt.ndim == 1, f"{batch_gt.ndim}"

    b = batch_img.shape[0]
    h, w = batch_img.shape[2:]

    le = 1
    out_image = np.zeros((h * le, w * b, 3), dtype="uint8")

    for i in range(b):
        img = batch_img[i]
        y_preds = batch_preds[i]

        img = img_denormalize_fn(img)
        img = tensor_to_numpy(img)
        pred_label = y_preds.cpu().item()

        target = f"p={pred_label}"

        if batch_gt is not None:
            gt_label = batch_gt[i]
            gt_label = gt_label.cpu().item()
            target += f" | gt={gt_label}"

        out_image[0:h, i * w : (i + 1) * w, :] = render_datapoint(img, target, text_size=12)

    return out_image
