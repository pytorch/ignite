from typing import Callable, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["Loss"]


class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is expected to be a tuple `(prediction, target)` or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments. If extra keywords arguments are provided they are passed to `loss_fn`.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).
        device (str or torch.device): specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    """

    required_output_keys = None

    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = lambda x: len(x),
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Loss, self).__init__(output_transform, device=device)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]) -> None:
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred.detach(), y.detach(), **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = self._batch_size(y)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> None:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")
        return self._sum.item() / self._num_examples
