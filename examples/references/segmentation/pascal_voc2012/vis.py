import numpy as np
import torch
from PIL import Image

try:
    from image_dataset_viz import render_datapoint
except ImportError:
    raise ModuleNotFoundError(
        "Please install image-dataset-viz via pip install --upgrade git+https://github.com/vfdev-5/ImageDatasetViz.git"
    )


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)


def render_mask(mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask.putpalette(vocpallete)
    mask = mask.convert(mode="RGB")
    return mask


def tensor_to_rgb(t):
    img = t.cpu().numpy().transpose((1, 2, 0))
    return img.astype(np.uint8)


def make_grid(batch_img, batch_mask, img_denormalize_fn, batch_gt_mask=None):
    """Create a grid from batch image and mask as

        img1  | img2  | img3  | img4  | ...
        i+m1  | i+m2  | i+m3  | i+m4  | ...
        mask1 | mask2 | mask3 | mask4 | ...
        i+M1  | i+M2  | i+M3  | i+M4  | ...
        Mask1 | Mask2 | Mask3 | Mask4 | ...

        i+m = image + mask blended with alpha=0.4
        - maskN is predicted mask
        - MaskN is ground-truth mask if given

    Args:
        batch_img (torch.Tensor) batch of images of any type
        batch_mask (torch.Tensor) batch of masks
        img_denormalize_fn (Callable): function to denormalize batch of images
        batch_gt_mask (torch.Tensor, optional): batch of ground truth masks.
    """
    assert isinstance(batch_img, torch.Tensor) and isinstance(batch_mask, torch.Tensor)
    assert len(batch_img) == len(batch_mask)

    if batch_gt_mask is not None:
        assert isinstance(batch_gt_mask, torch.Tensor)
        assert len(batch_mask) == len(batch_gt_mask)

    b = batch_img.shape[0]
    h, w = batch_img.shape[2:]

    le = 3 if batch_gt_mask is None else 3 + 2
    out_image = np.zeros((h * le, w * b, 3), dtype="uint8")

    for i in range(b):
        img = batch_img[i]
        mask = batch_mask[i]

        img = img_denormalize_fn(img)
        img = tensor_to_rgb(img)
        mask = mask.cpu().numpy()
        mask = render_mask(mask)

        out_image[0:h, i * w : (i + 1) * w, :] = img
        out_image[1 * h : 2 * h, i * w : (i + 1) * w, :] = render_datapoint(img, mask, blend_alpha=0.4)
        out_image[2 * h : 3 * h, i * w : (i + 1) * w, :] = mask

        if batch_gt_mask is not None:
            gt_mask = batch_gt_mask[i]
            gt_mask = gt_mask.cpu().numpy()
            gt_mask = render_mask(gt_mask)
            out_image[3 * h : 4 * h, i * w : (i + 1) * w, :] = render_datapoint(img, gt_mask, blend_alpha=0.4)
            out_image[4 * h : 5 * h, i * w : (i + 1) * w, :] = gt_mask

    return out_image


def predictions_gt_images_handler(img_denormalize_fn, n_images=None, another_engine=None, prefix_tag=None):
    def wrapper(engine, logger, event_name):
        batch = engine.state.batch
        output = engine.state.output
        x = batch["image"]
        y = batch["mask"]
        y_pred = output[0]

        if y.shape == y_pred.shape and y.ndim == 4:
            # Case of y of shape (B, C, H, W)
            y = torch.argmax(y, dim=1)

        y_pred = torch.argmax(y_pred, dim=1).byte()

        if n_images is not None:
            x = x[:n_images, ...]
            y = y[:n_images, ...]
            y_pred = y_pred[:n_images, ...]

        grid_pred_gt = make_grid(x, y_pred, img_denormalize_fn, batch_gt_mask=y)

        state = engine.state if another_engine is None else another_engine.state
        global_step = state.epoch

        tag = "predictions_with_gt"
        if prefix_tag is not None:
            tag = f"{prefix_tag}: {tag} - epoch={global_step}"
        logger.writer.add_image(tag=tag, img_tensor=grid_pred_gt, global_step=global_step, dataformats="HWC")

    return wrapper
