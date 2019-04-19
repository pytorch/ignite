from __future__ import division

import torch

from ignite.contrib.metrics.regression._base import _BaseRegressionEpoch


def r2_score_compute_fn(y_pred, y):
    e = torch.sum((y - y_pred) ** 2) / torch.sum((y - y.mean()) ** 2)
    return 1 - e.item()


class R2Score(_BaseRegressionEpoch):
    r"""
    Calculates the R-Squared:

    :math:`R^2 = 1 - \frac{\sum_{j=1}^n{(A_j - P_j)^2}}{\sum_{j=1}^n{(A_j - \bar{A})^2}}`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

    """
    def __init__(self, output_transform=lambda x: x):
        super(R2Score, self).__init__(r2_score_compute_fn, output_transform)
