from ignite.metrics._classification_support import PrecisionRecallSupport


class Precision(PrecisionRecallSupport):
    """
    Calculates precision.
    - | `threshold_function` is only needed for binary cases. Default is `torch.round(x)`. It is used to convert
      | `y_pred` to 0's and 1's.
    - `update` must receive output of the form `(y_pred, y)`.
    - | For binary or multiclass cases, `y_pred` must be in the following shape (batch_size, num_categories, ...) or
      | (batch_size, ...) and `y` must be in the following shape (batch_size, ...).
    For binary or multiclass cases, if `average` is True, returns the unweighted average across all classes.
    Otherwise, returns a tensor with the precision for each class.
    """
    def __init__(self, output_transform=lambda x: x, average=False, threshold_function=None):
        self._precision_vs_recall = True
        super(Precision, self).__init__(output_transform=output_transform, average=average,
                                        threshold_function=threshold_function)
