from typing import Any, Callable, Tuple, Union

import torch

from torch import Tensor

from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetric
from ignite.metrics.regression._base import _check_output_shapes, _check_output_types


def _get_spearman_r() -> Callable[[Tensor, Tensor], float]:
    from scipy.stats import spearmanr

    def _compute_spearman_r(predictions: Tensor, targets: Tensor) -> float:
        np_preds = predictions.flatten().numpy()
        np_targets = targets.flatten().numpy()
        r = spearmanr(np_preds, np_targets).statistic
        return r

    return _compute_spearman_r


class SpearmanRankCorrelation(EpochMetric):
    r"""Calculates the
    `Spearman's rank correlation coefficient
    <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_.

    .. math::
        r_\text{s} = \text{Corr}[R[P], R[A]] = \frac{\text{Cov}[R[P], R[A]]}{\sigma_{R[P]} \sigma_{R[A]}}

    where :math:`A` and :math:`P` are the ground truth and predicted value,
    and :math:`R[X]` is the ranking value of :math:`X`.

    The computation of this metric is implemented with
    `scipy.stats.spearmanr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html>`_.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    Parameters are inherited from ``Metric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
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
    """

    def __init__(
        self,
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        try:
            from scipy.stats import spearmanr  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        super().__init__(_get_spearman_r(), output_transform, check_compute_fn, device, skip_unrolling)

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(1)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        _check_output_shapes(output)
        _check_output_types(output)

        super().update(output)

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                "SpearmanRankCorrelation must have at least one example before it can be computed."
            )

        return super().compute()
