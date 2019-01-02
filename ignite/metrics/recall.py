import torch
from ignite.metrics.precision import _BasePrecisionRecall
from ignite._utils import to_onehot


class Recall(_BasePrecisionRecall):
    """
    Calculates recall for binary and multiclass data

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...)
    - `y` must be in the following shape (batch_size, ...)

    In binary case, when `y` has 0 or 1 values, the elements of `y_pred` must be between 0 and 1. Recall is
    computed over positive class, assumed to be 1.

    Args:
        average (bool, optional): if True, recall is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with the recall (for each class in multiclass case).

    """

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if self._type == "binary":
            y_pred = torch.round(y_pred).view(-1)
            y = y.view(-1)
        elif self._type == "multiclass":
            num_classes = y_pred.size(1)
            y = to_onehot(y.view(-1), num_classes=num_classes)
            indices = torch.max(y_pred, dim=1)[1].view(-1)
            y_pred = to_onehot(indices, num_classes=num_classes)

        y_pred = y_pred.type(dtype)
        y = y.type(dtype)
        correct = y * y_pred
        actual_positives = y.sum(dim=0)

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=0)

        self._true_positives += true_positives
        self._positives += actual_positives
