import numbers
from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.metrics.metrics_lambda import MetricsLambda

__all__ = ["ConfusionMatrix", "mIoU", "IoU", "DiceCoefficient", "cmAccuracy", "cmPrecision", "cmRecall"]


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): optional device specification for internal storage.

    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.

    """

    def __init__(
        self,
        num_classes: int,
        average: Optional[str] = None,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of 'samples', 'recall', 'precision'")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrix, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64, device=self._device)
        self._num_examples = 0

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...), " "but given {}".format(y_pred.shape)
            )

        if y_pred.shape[1] != self.num_classes:
            raise ValueError(
                "y_pred does not have correct number of categories: {} vs {}".format(y_pred.shape[1], self.num_classes)
            )

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...) and y must have "
                "shape of (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output

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
        if average not in ("recall", "precision"):
            raise ValueError("Argument average one of 'samples', 'recall', 'precision'")

        if average == "recall":
            return matrix / (matrix.sum(dim=1).unsqueeze(1) + 1e-15)
        elif average == "precision":
            return matrix / (matrix.sum(dim=0) + 1e-15)


def IoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates Intersection over Union using :class:`~ignite.metrics.ConfusionMatrix` metric.

    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index

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
        raise TypeError("Argument cm should be instance of ConfusionMatrix, but given {}".format(type(cm)))

    if not (cm.average in (None, "samples")):
        raise ValueError("ConfusionMatrix should have average attribute either None or 'samples'")

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError(
                    "ignore_index {} is larger than the length of IoU vector {}".format(ignore_index, len(iou_vector))
                )
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def mIoU(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates mean Intersection over Union using :class:`~ignite.metrics.ConfusionMatrix` metric.

    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index

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
    return IoU(cm=cm, ignore_index=ignore_index).mean()


def cmAccuracy(cm: ConfusionMatrix) -> MetricsLambda:
    """Calculates accuracy using :class:`~ignite.metrics.ConfusionMatrix` metric.

    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric

    Returns:
        MetricsLambda
    """
    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    return cm.diag().sum() / (cm.sum() + 1e-15)


def cmPrecision(cm: ConfusionMatrix, average: bool = True) -> MetricsLambda:
    """Calculates precision using :class:`~ignite.metrics.ConfusionMatrix` metric.

    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    if average:
        return precision.mean()
    return precision


def cmRecall(cm: ConfusionMatrix, average: bool = True) -> MetricsLambda:
    """
    Calculates recall using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    if average:
        return recall.mean()
    return recall


def DiceCoefficient(cm: ConfusionMatrix, ignore_index: Optional[int] = None) -> MetricsLambda:
    """Calculates Dice Coefficient for a given :class:`~ignite.metrics.ConfusionMatrix` metric.

    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index
    """

    if not isinstance(cm, ConfusionMatrix):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, but given {}".format(type(cm)))

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    dice = 2.0 * cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-15)

    if ignore_index is not None:

        def ignore_index_fn(dice_vector: torch.Tensor) -> torch.Tensor:
            if ignore_index >= len(dice_vector):
                raise ValueError(
                    "ignore_index {} is larger than the length of Dice vector {}".format(ignore_index, len(dice_vector))
                )
            indices = list(range(len(dice_vector)))
            indices.remove(ignore_index)
            return dice_vector[indices]

        return MetricsLambda(ignore_index_fn, dice)
    else:
        return dice
