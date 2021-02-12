import numbers
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.metrics.metrics_lambda import MetricsLambda

__all__ = [
    "MultiLabelConfusionMatrix",
]


class MultiLabelConfusionMatrix(Metric):
    def __init__(
        self,
        num_classes: int,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        normalized: bool = False,
    ):
        if num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

        self.num_classes = num_classes
        self._num_examples = 0
        self.normalized = normalized
        super(MultiLabelConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, 2, 2, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_input(output)
        y_pred, y = output[0].detach(), output[1].detach()

        self._num_examples += y.shape[0]
        y_reshaped = y.swapaxes(0, 1).reshape(self.num_classes, -1)
        y_pred_reshaped = y_pred.swapaxes(0, 1).reshape(self.num_classes, -1)
        y_total = torch.unsqueeze(torch.count_nonzero(y_reshaped, axis=1), 0)
        y_pred_total = torch.unsqueeze(torch.count_nonzero(y_pred_reshaped, axis=1), 0)
        tp = torch.unsqueeze(torch.count_nonzero(torch.mul(y_reshaped, y_pred_reshaped), axis=1), 0)
        fp = y_pred_total - tp
        fn = y_total - tp
        tn = y_reshaped.shape[1] - tp - fp - fn

        self.confusion_matrix += torch.cat([tn, fp, fn, tp], dim=0).T.reshape(-1, 2, 2)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")

        if self.normalized:
            conf = self.confusion_matrix.type(torch.float64)
            sums = conf.sum(axis=2).sum(axis=1)
            return (conf.reshape(conf.shape[0], 4) / sums[:, None]).reshape(conf.shape)
        else:
            return self.confusion_matrix

    def _check_input(self, output: Sequence[torch.Tensor]) -> None:
        if (
            not isinstance(output, Sequence)
            or len(output) < 2
            or not isinstance(output[0], torch.Tensor)
            or not isinstance(output[1], torch.Tensor)
        ):
            raise ValueError(
                (r"Argument must consist of a Python Sequence of two tensors such that the first is the predicted"
                 r" tensor and the second is the ground-truth tensor")
            )

        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() < 2:
            raise ValueError(
                f"y_pred must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y.ndimension() < 2:
            raise ValueError(
                f"y must at least have shape (batch_size, num_classes (currently set to {self.num_classes}), ...)"
            )

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"y_pred and y have different batch size: {y_pred.shape[0]} vs {y.shape[0]}")

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")

        if y.shape[1] != self.num_classes:
            raise ValueError(f"y does not have correct number of classes: {y.shape[1]} vs {self.num_classes}")

        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred shapes must match.")

        valid_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if y_pred.dtype not in valid_types:
            raise ValueError(f"y_pred must be of any type: {valid_types}")

        if y.dtype not in valid_types:
            raise ValueError(f"y must be of any type: {valid_types}")

        if y_pred.numel() != ((y_pred == 0).sum() + (y_pred == 1).sum()).item():
            raise ValueError("y_pred must be a binary tensor")

        if y.numel() != ((y == 0).sum() + (y == 1).sum()).item():
            raise ValueError("y must be a binary tensor")
