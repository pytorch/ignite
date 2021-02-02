from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError


class FractionalAbsoluteError(_BaseRegression):
    r"""Calculates the Fractional Absolute Error.

    .. math::
        \text{FAE} = \frac{1}{n}\sum_{j=1}^n\frac{2 |A_j - P_j|}{|A_j| + |P_j|}

    where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006
    """

    def reset(self) -> None:
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        errors = 2 * torch.abs(y.view_as(y_pred) - y_pred) / (torch.abs(y_pred) + torch.abs(y.view_as(y_pred)))
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError(
                "FractionalAbsoluteError must have at least one example before it can be computed."
            )
        return self._sum_of_errors / self._num_examples
