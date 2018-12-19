from __future__ import division
import torch
from ignite.contrib.metrics.regression._base import _BaseRegressionEpoch


def median_relative_absolute_error_compute_fn(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred) - torch.mean(y))
    return torch.median(e).item()


class MedianRelativeAbsoluteError(_BaseRegressionEpoch):
    r"""
    Calculates the Median Relative Absolute Error:

    :math:`\text{MdRAE} = \text{MD_{j=1,n}}\frac{|A_j - P_j|}{|A_j - \bar{A}|}`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.


    __ https://arxiv.org/abs/1809.03006

    """
    def __init__(self, output_transform=lambda x: x):
        super(MedianRelativeAbsoluteError, self).__init__(median_relative_absolute_error_compute_fn,
                                                          output_transform)
