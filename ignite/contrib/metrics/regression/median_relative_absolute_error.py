from typing import Callable, Union

import torch

from ignite.metrics import EpochMetric


def median_relative_absolute_error_compute_fn(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred) - torch.mean(y))
    return torch.median(e).item()


class MedianRelativeAbsoluteError(EpochMetric):
    r"""Calculates the Median Relative Absolute Error.

    .. math::
        \text{MdRAE} = \text{MD}_{j=1,n} \left( \frac{|A_j - P_j|}{|A_j - \bar{A}|} \right)

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.


    __ https://arxiv.org/abs/1809.03006

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: optional device specification for internal storage.
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ):
        super(MedianRelativeAbsoluteError, self).__init__(
            median_relative_absolute_error_compute_fn, output_transform=output_transform, device=device
        )
