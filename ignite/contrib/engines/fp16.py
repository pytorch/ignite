import torch
import torch.nn as nn

from ignite.engine import _prepare_batch, Engine


def create_supervised_fp16_trainer(model, optimizer, criterion, prepare_batch=_prepare_batch, **fp16_optimizer_kwargs):
    """
    Factory function for creating a trainer for supervised models using `half-precision floating-point`__.

    __ https://github.com/NVIDIA/apex/

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        criterion (torch.nn loss function): the loss function to use
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`
        **fp16_optimizer_kwargs: kwargs for `apex.fp16_utils.FP16_Optimizer`

    Returns:
        Engine: a trainer engine with supervised update function
    """

    try:
        from apex.fp16_utils import FP16_Optimizer, network_to_half
    except ImportError:
        raise RuntimeError("This contrib module requires NVidia/Apex to be installed")

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA should be available")
    if not torch.backends.cudnn.enabled:
        raise RuntimeError("NVidia/Apex fp16 mode requires cudnn backend to be enabled.")

    model = network_to_half(model.to(device))
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)

    if not fp16_optimizer_kwargs:
        fp16_optimizer_kwargs = {
            "verbose": False,
            "static_loss_scale": 128.0,
            "dynamic_loss_scale": False
        }

    optimizer = FP16_Optimizer(optimizer, **fp16_optimizer_kwargs)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=True)
        x = x.half()
        y = y.half()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.backward(loss)
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_fp16_evaluator(model, metrics={},
                                     device=None, non_blocking=False,
                                     prepare_batch=_prepare_batch):
    """
    Factory function for creating an evaluator for supervised models using `half-precision floating-point`__.

    __ https://github.com/NVIDIA/apex/

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (Callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.

    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            x = x.half()
            y = y.half()
            y_pred = model(x)
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
