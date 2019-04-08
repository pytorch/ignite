import numbers

import torch

from ignite.metrics import Metric, MetricsLambda
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` can be of two types:
        - shape (batch_size, num_categories, ...)
        - shape (batch_size, ...) and contains ground-truth class indices

    Args:
        num_classes (int): number of classes. In case of images, num_classes should also count the background index 0.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    """

    def __init__(self, num_classes, average=None, output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float)
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
                             "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}"
                             .format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...) and y must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

        return y_pred, y

    def update(self, output):
        y_pred, y = self._check_shape(output)

        if y_pred.shape != y.shape:
            y_ohe = to_onehot(y.reshape(-1), self.num_classes)
            y_ohe_t = y_ohe.transpose(0, 1).float()
        else:
            y_ohe_t = y.transpose(1, -1).reshape(y.shape[1], -1).float()

        indices = torch.argmax(y_pred, dim=1)
        y_pred_ohe = to_onehot(indices.reshape(-1), self.num_classes)
        y_pred_ohe = y_pred_ohe.float()

        if self.confusion_matrix.type() != y_ohe_t.type():
            self.confusion_matrix = self.confusion_matrix.type_as(y_ohe_t)

        self.confusion_matrix += torch.matmul(y_ohe_t, y_pred_ohe).float()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average:
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix.cpu()


def IoU(cm, ignore_index=None):
    """Calculates Intersection over Union

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

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision
    cm = cm.type(torch.float64)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError("ignore_index {} is larger than the length of IoU vector {}"
                                 .format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def mIoU(cm, ignore_index=None):
    """Calculates mean Intersection over Union

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


def cmAccuracy(cm):
    """
    Calculates accuracy using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric

    Returns:
        MetricsLambda
    """
    # Increase floating point precision
    cm = cm.type(torch.float64)
    return cm.diag().sum() / (cm.sum() + 1e-15)


def cmPrecision(cm, average=True):
    """
    Calculates precision using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision
    cm = cm.type(torch.float64)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    if average:
        return precision.mean()
    return precision


def cmRecall(cm, average=True):
    """
    Calculates recall using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision
    cm = cm.type(torch.float64)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    if average:
        return recall.mean()
    return recall
