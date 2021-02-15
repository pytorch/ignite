from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.engine.events import CallableEventWithFilter, EventEnum, Events, EventsList, RemovableEventHandle, State
from ignite.metrics import Metric
from ignite.utils import convert_tensor

if idist.has_xla_support:
    import torch_xla.core.xla_model as xm


__all__ = [
    "State",
    "create_supervised_trainer",
    "create_supervised_evaluator",
    "Engine",
    "DeterministicEngine",
    "Events",
    "EventsList",
    "EventEnum",
    "CallableEventWithFilter",
    "RemovableEventHandle",
    "supervised_trainer_step",
    "supervised_trainer_step_amp",
    "supervised_trainer_step_apex",
    "supervised_trainer_step_tpu",
]


def _prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )


def _arg_check(on_tpu: bool, amp_mode: Optional[str], scaler: Optional["torch.cuda.amp.GradScaler"]) -> None:
    """Checking tpu, amp and GradScaler instance combinations."""
    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    if amp_mode and on_tpu:
        raise ValueError("amp_mode cannot be used with xla device. Consider using amp_mode=None or device='cuda'.")

    if scaler is not None and not amp_mode:
        raise ValueError(
            "scaler argument is provided, but amp_mode is not. Please choose ('amp', 'apex') for amp_mode."
        )


def _train_zero_grad_prepare_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[Union[str, torch.device]],
    non_blocking: bool,
    prepare_batch: Callable,
    batch: Any,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Convert model to train mode, gradient to zero and get a batch."""
    model.train()
    optimizer.zero_grad()
    return prepare_batch(batch, device=device, non_blocking=non_blocking)


def _forward_loss(
    model: torch.nn.Module, loss_fn: Union[Callable, torch.nn.Module], x: Any, y: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Model forward pass and compute loss."""
    y_pred: torch.Tensor = model(x)
    loss: torch.Tensor = loss_fn(y_pred, y)
    return y_pred, loss


def supervised_trainer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]],
    non_blocking: bool,
    prepare_batch: Callable,
    output_transform: Callable,
) -> Callable:
    """Helper function defined the training step.

    Args:
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.
    """

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        x, y = _train_zero_grad_prepare_batch(model, optimizer, device, non_blocking, prepare_batch, batch)
        y_pred, loss = _forward_loss(model, loss_fn, x, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_trainer_step_amp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]],
    non_blocking: bool,
    prepare_batch: Callable,
    output_transform: Callable,
    scaler: Optional["torch.cuda.amp.GradScaler"],
) -> Callable:
    """Helper function defined the training step for torch.cuda.amp.

    Args:
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler instance for gradient scaling if `torch>=1.6.0`
            and ``amp_mode`` is ``amp``. If ``amp_mode`` is ``apex``, this argument will be ignored.

    Returns:
        Callable: update function.
    """

    if hasattr(torch.cuda.amp, "autocast"):
        from torch.cuda.amp import autocast
    else:
        raise AttributeError("autocast cannot be imported, please install torch>=1.6.0.")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        x, y = _train_zero_grad_prepare_batch(model, optimizer, device, non_blocking, prepare_batch, batch)
        with autocast():
            y_pred, loss = _forward_loss(model, loss_fn, x, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_trainer_step_apex(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]],
    non_blocking: bool,
    prepare_batch: Callable,
    output_transform: Callable,
) -> Callable:
    """Helper function defined the training step for apex.

    Args:
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.
    """

    try:
        from apex import amp as apex_amp
    except ImportError:
        raise ImportError("Please install apex from https://github.com/nvidia/apex to use amp_mode.")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        x, y = _train_zero_grad_prepare_batch(model, optimizer, device, non_blocking, prepare_batch, batch)
        y_pred, loss = _forward_loss(model, loss_fn, x, y)
        with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_trainer_step_tpu(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]],
    non_blocking: bool,
    prepare_batch: Callable,
    output_transform: Callable,
) -> Callable:
    """Helper function defined the training step for tpu.

    Args:
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.
    """

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        x, y = _train_zero_grad_prepare_batch(model, optimizer, device, non_blocking, prepare_batch, batch)
        y_pred, loss = _forward_loss(model, loss_fn, x, y)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        return output_transform(x, y, y_pred, loss)

    return update


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    deterministic: bool = False,
    amp_mode: Optional[str] = None,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Engine:
    """Factory function for creating a trainer for supervised models.

    Args:
        model (torch.nn.Module): the model to train.
        optimizer (torch.optim.Optimizer): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic (bool, optional): if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
            (default: False).
        amp_mode (str, optional): can be ``amp`` or ``apex``, model and optimizer will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ for ``amp`` and
            using `apex <https://nvidia.github.io/apex>`_ for ``apex``. (default: None)
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler instance for gradient scaling if `torch>=1.6.0`
            and ``amp_mode`` is ``amp``. If ``amp_mode`` is ``apex``, this argument will be ignored.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    .. warning::
        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. warning::
        If ``apex`` has been installed and torch version is less than 1.6.0, the model(s) and optimizer(s)
        must be initialized beforehand if ``amp_mode`` is provided since ``amp.initialize`` should be called
        after you have finished constructing your model(s) and optimizer(s), but before you send your model
        through any DistributedDataParallel wrapper.

        See more: https://nvidia.github.io/apex/amp.html#module-apex.amp

    Returns:
        Engine: a trainer engine with supervised update function.

    .. versionchanged:: 0.5.0

        - Added ``amp_mode`` argument for automatic mixed precision.
        - Added ``scaler`` argument for GradScaler instance.
    """

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    _arg_check(on_tpu, amp_mode, scaler)

    if amp_mode == "amp":
        _update = supervised_trainer_step_amp(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform, scaler
        )
    elif amp_mode == "apex":
        _update = supervised_trainer_step_apex(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )

    if on_tpu:
        _update = supervised_trainer_step_tpu(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )
    else:
        _update = supervised_trainer_step(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)

    return trainer


def create_supervised_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
