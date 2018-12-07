import torch
from ignite.metrics.precision import _BasePrecisionRecall
from ignite._utils import to_onehot


class Recall(_BasePrecisionRecall):
    """
    Calculates recall.
    - y_pred must be in the form of probabilities, use output_transform as needed.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    - In binary cases if a specific threshold probability is required, use output_transform.
    For binary cases, if `average` is True, returns precision of positive class, assumed to be 1.
    For multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.
    """

    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        dtype = y_pred.type()

        if y_pred.ndimension() == y.ndimension():
            y_pred = y_pred.unsqueeze(dim=1)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)

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
        if self._true_positives is None:
            self._true_positives = true_positives
            self._positives = actual_positives
        else:
            self._true_positives += true_positives
            self._positives += actual_positives
