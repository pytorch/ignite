import torch
from dataflow.vis import make_grid


def predictions_gt_images_handler(img_denormalize_fn, n_images=None, another_engine=None, prefix_tag=None):
    def wrapper(engine, logger, event_name):
        batch = engine.state.batch
        output = engine.state.output
        x, y = batch
        y_pred = output[0]

        if y.shape == y_pred.shape and y.ndim == 4:
            # Case of y of shape (B, C, H, W)
            y = torch.argmax(y, dim=1)

        y_pred = torch.argmax(y_pred, dim=1).byte()

        if n_images is not None:
            x = x[:n_images, ...]
            y = y[:n_images, ...]
            y_pred = y_pred[:n_images, ...]

        grid_pred_gt = make_grid(x, y_pred, img_denormalize_fn, batch_gt=y)

        state = engine.state if another_engine is None else another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        tag = "predictions_with_gt"
        if prefix_tag is not None:
            tag = f"{prefix_tag}: {tag}"
        logger.writer.add_image(tag=tag, img_tensor=grid_pred_gt, global_step=global_step, dataformats="HWC")

    return wrapper
