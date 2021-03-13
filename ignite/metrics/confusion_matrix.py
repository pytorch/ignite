import numbers
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.metrics.metrics_lambda import MetricsLambda

__all__ = ["ConfusionMatrix", "mIoU", "IoU", "DiceCoefficient", "cmAccuracy", "cmPrecision", "cmRecall", "JaccardIndex"]


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must contain logits and has the following shape (batch_size, num_classes, ...).
      If you are doing binary classification, see Note for an example on how to get this.
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
      with or without the background class. During the computation, argmax of `y_pred` is taken to determine
      predicted classes.

    Args:
        num_classes: Number of classes, should be > 1. See notes for more details.
        average: confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.

        If you are doing binary classification with a single output unit, you may have to transform your network output,
        so that you have one value for each class. E.g. you can transform your network output into a one-hot vector
        with:

        .. code-block:: python

            def binary_one_hot_output_transform(output):
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred).round().long()
                y_pred = ignite.utils.to_onehot(y_pred, 2)
                y = y.long()
                return y_pred, y

            metrics = {
                "confusion_matrix": ConfusionMatrix(2, output_transform=binary_one_hot_output_transform),
            }

            evaluator = create_supervised_evaluator(
                model, metrics=metrics, output_transform=lambda x, y, y_pred: (y_pred, y)
            )

    """

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of 'samples', 'recall', 'precision'")

        if num_classes <= 1:
            raise ValueError("Argument num_classes needs to be > 1")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        super(ConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() < 2:
            raise ValueError(
                f"y_pred must have shape (batch_size, num_classes (currently set to {self.num_classes}), ...), "
                f"but given {y_pred.shape}"
            )

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[1]} vs {self.num_classes}")

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                f"y_pred must have shape (batch_size, num_classes (currently set to {self.num_classes}), ...) "
                "and y must have shape of (batch_size, ...), "
                f"but given {y.shape} vs {y_pred.shape}."
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape  # type: Tuple[int, ...]

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    @sync_all_reduce("confusion_matrix", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            raise NotComputableError("Confusion matrix must have at least one example before it can be computed.")
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            else:
                return self.normalize(self.confusion_matrix, self.average)
        return self.confusion_matrix

    @staticmethod
    def normalize(matrix: torch.Tensor, average: str) -> torch.Tensor:
        """Normalize given `matrix` with given `average`."""
        if average == "recall":
            return matrix / (matrix.sum(dim=1).unsqueeze(1) + 1e-15)
        elif average == "precision":
            return matrix / (matrix.sum(dim=0) + 1e-15)
        else:
            raise ValueError("Argument average should be one of 'samples', 'recall', 'precision'")


def IoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    r"""Calculates Intersection over Union using :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` metric.

    .. math:: \text{J}(A, B) = \frac{ \lvert A \cap B \rvert }{ \lvert A \cup B \rvert }

    Args:
        cm: instance of confusion matrix metric
        ignore_index: index to ignore, e.g. background index

    Returns:
        MetricsLambda

    Examples:

    .. code-block:: python

        train_evaluator = ...

        cm = ConfusionMatrix(num_classes=num_classes)
        IoU(cm, ignore_index=0).attach(train_evaluator, 'IoU')

        state = train_evaluator.run(train_dataset)
        # state.metrics['IoU'] -> tensor of shape (num_classes - 1, )

    """
    if not isinstance(cm, ConfusionMatrix):
        raise TypeError(f"Argument cm should be instance of ConfusionMatrix, but given {type(cm)}")

    if not (cm.average in (None, "samples")):
        raise ValueError("ConfusionMatrix should have average attribute either None or 'samples'")

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError(f"ignore_index should be non-negative integer, but given {ignore_index}")

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)  # type: MetricsLambda
    if ignore_index is not None:
        ignore_idx = ignore_index  # type: int  # used due to typing issues with mympy

        def ignore_index_fn(iou_vector: torch.Tensor) -> torch.Tensor:
            if ignore_idx >= len(iou_vector):
                raise ValueError(f"ignore_index {ignore_idx} is larger than the length of IoU vector {len(iou_vector)}")
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_idx)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def mIoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates mean Intersection over Union using :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` metric.

    Args:
        cm: instance of confusion matrix metric
        ignore_index: index to ignore, e.g. background index

    Returns:
        MetricsLambda

    Examples:

    .. code-block:: python

        train_evaluator = ...

        cm = ConfusionMatrix(num_classes=num_classes)
        mIoU(cm, ignore_index=0).attach(train_evaluator, 'mean IoU')

        state = train_evaluator.run(train_dataset)
        # state.metrics['mean IoU'] -> scalar


    """
    iou = IoU(cm=cm, ignore_index=ignore_index).mean()  # type: MetricsLambda
    return iou


def cmAccuracy(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates accuracy using :class:`~ignite.metrics.metric.ConfusionMatrix` metric.

    Args:
        cm: instance of confusion matrix metric

    Returns:
        MetricsLambda
    """
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    accuracy = cm.diag().sum() / (cm.sum() + 1e-15)  # type: MetricsLambda
    return accuracy


def cmPrecision(cm: ConfusionMatrix, average: bool = True) -> MetricsLambda:
    """Calculates precision using :class:`~ignite.metrics.metric.ConfusionMatrix` metric.

    Args:
        cm: instance of confusion matrix metric
        average: if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)  # type: MetricsLambda
    if average:
        mean = precision.mean()  # type: MetricsLambda
        return mean
    return precision


def cmRecall(cm: ConfusionMatrix, average: bool = True) -> MetricsLambda:
    """
    Calculates recall using :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` metric.
    Args:
        cm: instance of confusion matrix metric
        average: if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)  # type: MetricsLambda
    if average:
        mean = recall.mean()  # type: MetricsLambda
        return mean
    return recall


def DiceCoefficient(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates Dice Coefficient for a given :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` metric.

    Args:
        cm: instance of confusion matrix metric
        ignore_index: index to ignore, e.g. background index
    """

    if not isinstance(cm, ConfusionMatrix):
        raise TypeError(f"Argument cm should be instance of ConfusionMatrix, but given {type(cm)}")

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError(f"ignore_index should be non-negative integer, but given {ignore_index}")

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    dice = 2.0 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)  # type: MetricsLambda

    if ignore_index is not None:
        ignore_idx = ignore_index  # type: int  # used due to typing issues with mympy

        def ignore_index_fn(dice_vector: torch.Tensor) -> torch.Tensor:
            if ignore_idx >= len(dice_vector):
                raise ValueError(
                    f"ignore_index {ignore_idx} is larger than the length of Dice vector {len(dice_vector)}"
                )
            indices = list(range(len(dice_vector)))
            indices.remove(ignore_idx)
            return dice_vector[indices]

        return MetricsLambda(ignore_index_fn, dice)
    else:
        return dice


def JaccardIndex(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    r"""Calculates the Jaccard Index using :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` metric.
    Implementation is based on :meth:`~ignite.metrics.IoU`.

    .. math:: \text{J}(A, B) = \frac{ \lvert A \cap B \rvert }{ \lvert A \cup B \rvert }

    Args:
        cm: instance of confusion matrix metric
        ignore_index: index to ignore, e.g. background index

    Returns:
        MetricsLambda

    Examples:

    .. code-block:: python

        train_evaluator = ...

        cm = ConfusionMatrix(num_classes=num_classes)
        JaccardIndex(cm, ignore_index=0).attach(train_evaluator, 'JaccardIndex')

        state = train_evaluator.run(train_dataset)
        # state.metrics['JaccardIndex'] -> tensor of shape (num_classes - 1, )

    """
    return IoU(cm, ignore_index)
