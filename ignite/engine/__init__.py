from typing import Sequence, Union, Optional, Callable, Dict, Any, Tuple

import torch

from ignite.engine.engine import Engine
from ignite.engine.events import State, Events
from ignite.utils import convert_tensor
from ignite.metrics import Metric

__all__ = [
    'create_supervised_trainer',
    'create_supervised_evaluator',
    'Engine',
    'Events'
]


def _prepare_batch(batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None,
                   non_blocking: bool = False):
    """Prepare batch for training: pass to a device with options.

    """
    return convert_tensor(batch, device=device, non_blocking=non_blocking)


def create_supervised_trainer(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                              loss_fn: Union[Callable, torch.nn.Module],
                              device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False,
                              prepare_batch: Callable = _prepare_batch,
                              output_transform: Callable = lambda x, y, y_pred, loss: loss.item(),
                              get_data: Callable = lambda batch: (batch[0], batch[1])) -> Engine:
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        get_data (callable) : function that receives a batch (as a mapping or a sequence) and returns 'x' and 'y'

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Note: The type of the batch received by `get_data` is the same as returned by the `__getitem__` of the used
        dataset but with batched tensors

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = get_data(prepare_batch(batch, device=device, non_blocking=non_blocking))
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def create_supervised_evaluator(model: torch.nn.Module, metrics: Optional[Dict[str, Metric]] = None,
                                device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False,
                                prepare_batch: Callable = _prepare_batch,
                                output_transform: Callable = lambda x, y, y_pred: (y_pred, y,),
                                get_data: Callable = lambda batch: (batch[0], batch[1])) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.
        get_data (callable) : function that receives a batch (as a mapping or a sequence) and returns 'x' and 'y'

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Note: The type of the batch received by `get_data` is the same as returned by the `__getitem__` of the used
        dataset but with batched tensors

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        with torch.no_grad():
            x, y = get_data(prepare_batch(batch, device=device, non_blocking=non_blocking))
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
