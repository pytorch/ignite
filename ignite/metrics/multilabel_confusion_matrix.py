from typing import Callable, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MultiLabelConfusionMatrix"]


class MultiLabelConfusionMatrix(Metric):
    """Calculates a confusion matrix for multi-labelled, multi-class data.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must contain 0s and 1s and has the following shape (batch_size, num_classes, ...).
      For example, `y_pred[i, j]` = 1 denotes that the j'th class is one of the labels of the i'th sample as predicted.
    - `y` should have the following shape (batch_size, num_classes, ...) with 0s and 1s. For example,
      `y[i, j]` = 1 denotes that the j'th class is one of the labels of the i'th sample according to the ground truth.
    - both `y` and `y_pred` must be torch Tensors having any of the following types:
      {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}. They must have the same dimensions.
    - The confusion matrix 'M' is of dimension (num_classes, 2, 2).

      * M[i, 0, 0] corresponds to count/rate of true negatives of class i
      * M[i, 0, 1] corresponds to count/rate of false positives of class i
      * M[i, 1, 0] corresponds to count/rate of false negatives of class i
      * M[i, 1, 1] corresponds to count/rate of true positives of class i

    - The classes present in M are indexed as 0, ... , num_classes-1 as can be inferred from above.

    Args:
        num_classes: Number of classes, should be > 1.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.
        normalized: whether to normalize confusion matrix by its sum or not.

    Example:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MultiLabelConfusionMatrix(num_classes=3)
            metric.attach(default_evaluator, "mlcm")
            y_true = torch.tensor([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
            ])
            y_pred = torch.tensor([
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["mlcm"])

        .. testoutput::

            tensor([[[0, 4],
                     [0, 1]],

                    [[3, 1],
                     [0, 1]],

                    [[1, 2],
                     [2, 0]]])

    .. versionadded:: 0.4.5

    """

    _state_dict_all_req_keys = ("confusion_matrix", "_num_examples")

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
        y_reshaped = y.transpose(0, 1).reshape(self.num_classes, -1)
        y_pred_reshaped = y_pred.transpose(0, 1).reshape(self.num_classes, -1)

        y_total = y_reshaped.sum(dim=1)
        y_pred_total = y_pred_reshaped.sum(dim=1)

        tp = (y_reshaped * y_pred_reshaped).sum(dim=1)
        fp = y_pred_total - tp
        fn = y_total - tp
        tn = y_reshaped.shape[1] - tp - fp - fn

        self.confusion_matrix += torch.stack([tn, fp, fn, tp], dim=1).reshape(-1, 2, 2).to(self._device)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")

        if self.normalized:
            conf = self.confusion_matrix.to(dtype=torch.float64)
            sums = conf.sum(dim=(1, 2))
            return conf / sums[:, None, None]

        return self.confusion_matrix

    def _check_input(self, output: Sequence[torch.Tensor]) -> None:
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

        if not torch.equal(y_pred, y_pred**2):
            raise ValueError("y_pred must be a binary tensor")

        if not torch.equal(y, y**2):
            raise ValueError("y must be a binary tensor")
