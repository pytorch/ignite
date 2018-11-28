import torch
from ignite.metrics import EpochMetric


def median_absolute_error_compute_fn(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred)
    return torch.median(e).item()


class MedianAbsoluteError(EpochMetric):
    r"""
    Calculates the Median Absolute Error:

    :math:`\text{MdAE} = \text{MD_{j=1,n}}{|A_j - P_j|}`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.


    __ https://arxiv.org/abs/1809.03006

    """
    def __init__(self, output_transform=lambda x: x):
        super(MedianAbsoluteError, self).__init__(median_absolute_error_compute_fn, output_transform)
