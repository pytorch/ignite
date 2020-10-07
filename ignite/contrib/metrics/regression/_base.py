from abc import abstractmethod
from typing import Callable, Tuple

import torch

from ignite.metrics import EpochMetric, Metric
from ignite.metrics.metric import reinit__is_reduced


def _check_output_shapes(output: Tuple[torch.Tensor, torch.Tensor]):
    y_pred, y = output
    if y_pred.shape != y.shape:
        raise ValueError("Input data shapes should be the same, but given {} and {}".format(y_pred.shape, y.shape))

    c1 = y_pred.ndimension() == 2 and y_pred.shape[1] == 1
    if not (y_pred.ndimension() == 1 or c1):
        raise ValueError("Input y_pred should have shape (N,) or (N, 1), but given {}".format(y_pred.shape))

    c2 = y.ndimension() == 2 and y.shape[1] == 1
    if not (y.ndimension() == 1 or c2):
        raise ValueError("Input y should have shape (N,) or (N, 1), but given {}".format(y.shape))


def _check_output_types(output: Tuple[torch.Tensor, torch.Tensor]):
    y_pred, y = output
    if y_pred.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("Input y_pred dtype should be float 16, 32 or 64, but given {}".format(y_pred.dtype))

    if y.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("Input y dtype should be float 16, 32 or 64, but given {}".format(y.dtype))


class _BaseRegression(Metric):
    # Base class for all regression metrics
    # `update` method check the shapes and call internal overloaded
    # method `_update`.

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        _check_output_shapes(output)
        _check_output_types(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        self._update((y_pred, y))

    @abstractmethod
    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        pass


class _BaseRegressionEpoch(EpochMetric):
    # Base class for all median-based regression metrics
    # `update` method check the shapes and call internal overloaded method `_update`.
    # Class internally stores complete history of predictions and targets of type float32.

    def __init__(
        self, compute_fn: Callable, output_transform: Callable = lambda x: x, check_compute_fn: bool = True,
    ):
        super(_BaseRegressionEpoch, self).__init__(
            compute_fn=compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )

    def _check_type(self, output: Tuple[torch.Tensor, torch.Tensor]):
        _check_output_types(output)
        super(_BaseRegressionEpoch, self)._check_type(output)

    def _check_shape(self, output: Tuple[torch.Tensor, torch.Tensor]):
        _check_output_shapes(output)
