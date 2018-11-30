from ignite.metrics.precision import _BasePrecisionRecallSupport


class Recall(_BasePrecisionRecallSupport):
    """
    Calculates recall.
    - | `threshold_function` is only needed for binary cases. Default is `torch.round(x)`. It is used to convert
      | `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the recall for each class.
    """
    def __init__(self, output_transform=lambda x: x, average=False, threshold_function=None):
        self._precision_vs_recall = False
        super(Recall, self).__init__(output_transform=output_transform, average=average,
                                     threshold_function=threshold_function)

    def update(self, output):
        correct, _, y = self._calculate_correct(output)
        actual_positives = y.sum(dim=0)
        self._sum_positives(correct, actual_positives)
