from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

import ignite.distributed as idist
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.engine.events import CallableEventWithFilter, EventEnum, Events, EventsList, RemovableEventHandle, State
from ignite.metrics import Metric
from ignite.utils import convert_tensor

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
    "supervised_training_step",
    "supervised_training_step_amp",
    "supervised_training_step_apex",
    "supervised_training_step_tpu",
    "supervised_evaluation_step",
    "supervised_evaluation_step_amp",
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


def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
) -> Callable:
    """Factory function for supervised training.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.

    Example::

        from ignite.engine import Engine, supervised_training_step

        model = ...
        optimizer = ...
        loss_fn = ...

        update_fn = supervised_training_step(model, optimizer, loss_fn, 'cuda')
        trainer = Engine(update_fn)

    .. versionadded:: 0.4.5
    """

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_training_step_amp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> Callable:
    """Factory function for supervised training using ``torch.cuda.amp``.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        scaler: GradScaler instance for gradient scaling. (default: None)

    Returns:
        Callable: update function

    Example::

        from ignite.engine import Engine, supervised_training_step_amp

        model = ...
        optimizer = ...
        loss_fn = ...
        scaler = torch.cuda.amp.GradScaler(2**10)

        update_fn = supervised_training_step_amp(model, optimizer, loss_fn, 'cuda', scaler=scaler)
        trainer = Engine(update_fn)

    .. versionadded:: 0.4.5
    """

    try:
        from torch.cuda.amp import autocast
    except ImportError:
        raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        with autocast(enabled=True):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_training_step_apex(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
) -> Callable:
    """Factory function for supervised training using apex.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.

    Example::

        from ignite.engine import Engine, supervised_training_step_apex

        model = ...
        optimizer = ...
        loss_fn = ...

        update_fn = supervised_training_step_apex(model, optimizer, loss_fn, 'cuda')
        trainer = Engine(update_fn)

    .. versionadded:: 0.4.5
    """

    try:
        from apex import amp as apex_amp
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please install apex from https://github.com/nvidia/apex to use amp_mode='apex'.")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        with apex_amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update


def supervised_training_step_tpu(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
) -> Callable:
    """Factory function for supervised training using ``torch_xla``.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, TPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Callable: update function.

    Example::

        from ignite.engine import Engine, supervised_training_step_tpu

        model = ...
        optimizer = ...
        loss_fn = ...

        update_fn = supervised_training_step_tpu(model, optimizer, loss_fn, 'xla')
        trainer = Engine(update_fn)

    .. versionadded:: 0.4.5
    """
    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError:
        raise ModuleNotFoundError("torch_xla cannot be imported, please install PyTorch XLA.")

    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        return output_transform(x, y, y_pred, loss)

    return update


def _check_arg(
    on_tpu: bool, amp_mode: Optional[str], scaler: Optional[Union[bool, "torch.cuda.amp.GradScaler"]]
) -> Tuple[Optional[str], Optional["torch.cuda.amp.GradScaler"]]:
    """Checking tpu, amp and GradScaler instance combinations."""
    if on_tpu and not idist.has_xla_support:
        raise RuntimeError("In order to run on TPU, please install PyTorch XLA")

    if amp_mode and on_tpu:
        raise ValueError("amp_mode cannot be used with xla device. Consider using amp_mode=None or device='cuda'.")

    if scaler:
        if amp_mode != "amp":
            raise ValueError(f"scaler argument is {scaler}, but amp_mode is {amp_mode}. Consider using amp_mode='amp'.")
        elif amp_mode == "amp" and isinstance(scaler, bool):
            try:
                from torch.cuda.amp import GradScaler
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use scaler argument.")
            scaler = GradScaler(enabled=True)

    if on_tpu:
        return "tpu", None
    elif scaler and amp_mode == "amp":
        return amp_mode, scaler  # type: ignore[return-value]
    else:
        return amp_mode, None


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
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
) -> Engine:
    """Factory function for creating a trainer for supervised models.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic: if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
            (default: False).
        amp_mode: can be ``amp`` or ``apex``, model and optimizer will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ for ``amp`` and
            using `apex <https://nvidia.github.io/apex>`_ for ``apex``. (default: None)
        scaler: GradScaler instance for gradient scaling if `torch>=1.6.0`
            and ``amp_mode`` is ``amp``. If ``amp_mode`` is ``apex``, this argument will be ignored.
            If True, will create default GradScaler. If GradScaler instance is passed, it will be used instead.
            (default: False)

    Returns:
        a trainer engine with supervised update function.

    Note:
        If ``scaler`` is True, GradScaler instance will be created internally and trainer state has attribute named
        ``scaler`` for that instance and can be used for saving and loading.

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
        If ``amp_mode='apex'`` , the model(s) and optimizer(s) must be initialized beforehand
        since ``amp.initialize`` should be called after you have finished constructing your model(s)
        and optimizer(s), but before you send your model through any DistributedDataParallel wrapper.

        See more: https://nvidia.github.io/apex/amp.html#module-apex.amp

    .. versionchanged:: 0.4.5

        - Added ``amp_mode`` argument for automatic mixed precision.
        - Added ``scaler`` argument for gradient scaling.
    """

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, amp_mode, scaler)

    if mode == "amp":
        _update = supervised_training_step_amp(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform, _scaler
        )
    elif mode == "apex":
        _update = supervised_training_step_apex(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )
    elif mode == "tpu":
        _update = supervised_training_step_tpu(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )
    else:
        _update = supervised_training_step(
            model, optimizer, loss_fn, device, non_blocking, prepare_batch, output_transform
        )

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
    if _scaler and scaler and isinstance(scaler, bool):
        trainer.state.scaler = _scaler  # type: ignore[attr-defined]

    return trainer


def supervised_evaluation_step(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Callable:
    """
    Factory function for supervised evaluation.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Returns:
        Inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

    .. versionadded:: 0.4.5
    """

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    return evaluate_step


def supervised_evaluation_step_amp(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
) -> Callable:
    """
    Factory function for supervised evaluation using ``torch.cuda.amp``.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Returns:
        Inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

    .. versionadded:: 0.4.5
    """
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            with autocast(enabled=True):
                y_pred = model(x)
            return output_transform(x, y, y_pred)

    return evaluate_step


def create_supervised_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = lambda x, y, y_pred: (y_pred, y),
    amp_mode: Optional[str] = None,
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model: the model to train.
        metrics: a map of metric names to Metrics.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.
        amp_mode: can be ``amp``, model will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_

    Returns:
        an evaluator engine with supervised inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. versionchanged:: 0.4.5
        - Added ``amp_mode`` argument for automatic mixed precision.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    mode, _ = _check_arg(on_tpu, amp_mode, None)

    metrics = metrics or {}
    if mode == "amp":
        evaluate_step = supervised_evaluation_step_amp(model, device, non_blocking, prepare_batch, output_transform)
    else:
        evaluate_step = supervised_evaluation_step(model, device, non_blocking, prepare_batch, output_transform)

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
