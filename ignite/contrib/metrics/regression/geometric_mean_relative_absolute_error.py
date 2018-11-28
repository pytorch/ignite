from __future__ import division
import torch
from ignite.metrics import EpochMetric


def geometric_mean_relative_absolute_error_compute_fn(y_pred, y):
    n = y_pred.shape[0]
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred) - torch.mean(y))
    return torch.pow(torch.prod(e), 1.0 / n)


class GeometricMeanRelativeAbsoluteError(EpochMetric):
    r"""
    Calculates the Geometric Mean Relative Absolute Error:

    :math:`\text{GMRAE} = TODO`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.


    __ https://arxiv.org/abs/1809.03006

    """
    def __init__(self, output_transform=lambda x: x):
        super(GeometricMeanRelativeAbsoluteError, self).__init__(geometric_mean_relative_absolute_error_compute_fn,
                                                                 output_transform)
