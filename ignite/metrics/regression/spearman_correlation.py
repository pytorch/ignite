from typing import Any
from collections.abc import Callable

import torch

from torch import Tensor

from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetric
from ignite.metrics.regression._base import _check_output_shapes, _check_output_types


def _get_ranks(x: Tensor) -> Tensor:
    """Calculates ranks with average method for ties natively in PyTorch."""
    n = x.size(0)
    # Get sorted indices and the inverse mapping
    sorter = torch.argsort(x)
    inv_sorter = torch.empty(n, dtype=torch.long, device=x.device)
    inv_sorter[sorter] = torch.arange(n, device=x.device)

    x_sorted = x[sorter]
    # Find ties
    obs = torch.cat([torch.tensor([True], device=x.device), x_sorted[1:] != x_sorted[:-1]])
    dense_ranks = torch.cumsum(obs, dim=0)

    # Calculate average ranks for ties
    count = torch.cat([torch.nonzero(obs).flatten(), torch.tensor([n], device=x.device)])
    repetitions = count[1:] - count[:-1]

    # Use cumsum of repetitions to find the range of ranks for each unique value
    right = torch.cumsum(repetitions, dim=0)
    left = right - repetitions + 1
    avg_ranks = (left + right).double() / 2.0

    # Map back to original order
    return avg_ranks[dense_ranks - 1][inv_sorter]


def _spearman_r(predictions: Tensor, targets: Tensor) -> float:
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()

    if torch.isnan(preds_flat).any() or torch.isnan(targets_flat).any():
        return float("nan")

    # Native PyTorch Ranking
    r_preds = _get_ranks(preds_flat)
    r_targets = _get_ranks(targets_flat)

    # Correlation of ranks (Pearson Correlation)
    mu_x = torch.mean(r_preds)
    mu_y = torch.mean(r_targets)

    diff_x = r_preds - mu_x
    diff_y = r_targets - mu_y

    norm_x = torch.norm(diff_x, 2)
    norm_y = torch.norm(diff_y, 2)

    if norm_x == 0 or norm_y == 0:
        return float("nan")

    r = torch.sum(diff_x * diff_y) / (norm_x * norm_y)
    return r.item()


class SpearmanRankCorrelation(EpochMetric):
    r"""Calculates the
    `Spearman's rank correlation coefficient
    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_.

    .. math::
        r_\text{s} = \text{Corr}[R[P], R[A]] = \frac{\text{Cov}[R[P], R[A]]}{\sigma_{R[P]} \sigma_{R[A]}}

    where :math:`A` and :math:`P` are the ground truth and predicted value,
    and :math:`R[X]` is the ranking value of :math:`X`.

    The computation of this metric is implemented natively in PyTorch.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    Parameters are inherited from ``Metric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        check_compute_fn: if True, ``compute_fn`` is run on the first batch of data to ensure there are no
            issues. If issues exist, user is warned that there might be an issue with the ``compute_fn``.
            Default, True.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SpearmanRankCorrelation()
            metric.attach(default_evaluator, 'spearman_corr')
            y_true = torch.tensor([0., 1., 2., 3., 4., 5.])
            y_pred = torch.tensor([0.5, 2.8, 1.9, 1.3, 6.0, 4.1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['spearman_corr'])

        .. testoutput::

            0.7142857142857143

    .. versionadded:: 0.5.2

    .. versionchanged:: 0.5.5
        Implementation updated to use a native PyTorch computation for rank calculation and
        correlation, removing the dependency on SciPy.
    """

    def __init__(
        self,
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        super().__init__(_spearman_r, output_transform, check_compute_fn, device, skip_unrolling)

    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(1)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        _check_output_shapes((y_pred, y))
        _check_output_types((y_pred, y))

        super().update((y_pred, y))

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                "SpearmanRankCorrelation must have at least one example before it can be computed."
            )

        return super().compute()
