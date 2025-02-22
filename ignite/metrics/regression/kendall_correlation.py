from typing import Any, Callable, Tuple, Union

import torch

from torch import Tensor

from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetric
from ignite.metrics.regression._base import _check_output_shapes, _check_output_types


def _get_kendall_tau(variant: str = "b") -> Callable[[Tensor, Tensor], float]:
    from scipy.stats import kendalltau

    if variant not in ("b", "c"):
        raise ValueError(f"variant accepts 'b' or 'c', got {variant!r}.")

    def _tau(predictions: Tensor, targets: Tensor) -> float:
        np_preds = predictions.flatten().cpu().numpy()
        np_targets = targets.flatten().cpu().numpy()
        r = kendalltau(np_preds, np_targets, variant=variant).statistic
        return r

    return _tau


class KendallRankCorrelation(EpochMetric):
    r"""Calculates the
    `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_.

    .. math::
        \tau = 1-\frac{2(\text{number of discordant pairs})}{\left( \begin{array}{c}n\\2\end{array} \right)}

    Two prediction-target pairs :math:`(P_i, A_i)` and :math:`(P_j, A_j)`, where :math:`i<j`,
    are said to be concordant when both :math:`P_i<P_j` and :math:`A_i<A_j` holds
    or both :math:`P_i>P_j` and :math:`A_i>A_j`.

    The `number of discordant pairs` counts the number of pairs that are not concordant.

    The computation of this metric is implemented with
    `scipy.stats.kendalltau <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html>`_.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    Parameters are inherited from ``Metric.__init__``.

    Args:
        variant: variant of kendall rank correlation. ``'b'`` or ``'c'`` is accepted.
            Details can be found
            `here <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient#Accounting_for_ties>`_.
            Default: ``'b'``
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
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = KendallRankCorrelation()
            metric.attach(default_evaluator, 'kendall_tau')
            y_true = torch.tensor([0., 1., 2., 3., 4., 5.])
            y_pred = torch.tensor([0.5, 2.8, 1.9, 1.3, 6.0, 4.1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['kendall_tau'])

        .. testoutput::

            0.4666666666666666

    .. versionadded:: 0.5.2
    """

    def __init__(
        self,
        variant: str = "b",
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        try:
            from scipy.stats import kendalltau  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        super().__init__(_get_kendall_tau(variant), output_transform, check_compute_fn, device, skip_unrolling)

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
            raise NotComputableError("KendallRankCorrelation must have at least one example before it can be computed.")

        return super().compute()
