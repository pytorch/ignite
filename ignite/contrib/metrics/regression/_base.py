from abc import abstractmethod
from typing import Tuple

import torch

from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced


def _check_output_shapes(output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    y_pred, y = output
    c1 = y_pred.ndimension() == 2 and y_pred.shape[1] == 1
    if not (y_pred.ndimension() == 1 or c1):
        raise ValueError(f"Input y_pred should have shape (N,) or (N, 1), but given {y_pred.shape}")

    c2 = y.ndimension() == 2 and y.shape[1] == 1
    if not (y.ndimension() == 1 or c2):
        raise ValueError(f"Input y should have shape (N,) or (N, 1), but given {y.shape}")

    if y_pred.shape != y.shape:
        raise ValueError(f"Input data shapes should be the same, but given {y_pred.shape} and {y.shape}")


def _check_output_types(output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    y_pred, y = output
    if y_pred.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"Input y_pred dtype should be float 16, 32 or 64, but given {y_pred.dtype}")

    if y.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError(f"Input y dtype should be float 16, 32 or 64, but given {y.dtype}")


class _BaseRegression(Metric):
    # Base class for all regression metrics
    # `update` method check the shapes and call internal overloaded
    # method `_update`.

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        _check_output_shapes(output)
        _check_output_types(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        self._update((y_pred, y))

    @abstractmethod
    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        pass
