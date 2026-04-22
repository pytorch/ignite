from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.epoch_metric import EpochMetric
from ignite.metrics.metric import reinit__is_reduced


def _kappa_from_conf(conf: torch.Tensor, weights: Literal["linear", "quadratic"] | None) -> float:
    n = conf.sum()
    if n == 0:
        raise NotComputableError("CohenKappa cannot be computed on an empty confusion matrix (n == 0).")

    n_classes = conf.shape[0]

    if weights is None:
        p_o = conf.trace() / n
        row = conf.sum(dim=1)
        col = conf.sum(dim=0)
        p_e = (row * col).sum() / (n * n)
    else:
        idx = torch.arange(n_classes, device=conf.device)
        if weights == "linear":
            w = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).double()
        else:
            w = ((idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2).double()

        w = w / w.max()
        p_o = 1 - (w * conf).sum() / n
        row = conf.sum(dim=1)
        col = conf.sum(dim=0)
        expected = row.unsqueeze(1) * col.unsqueeze(0) / n
        p_e = 1 - (w * expected).sum() / n

    if (1 - p_e).abs() < 1e-9:
        return 1.0 if (p_o - p_e).abs() < 1e-9 else float("nan")

    return ((p_o - p_e) / (1 - p_e)).item()


def _cohen_kappa_score(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    weights: Literal["linear", "quadratic"] | None,
) -> float:
    num_classes = int(max(y_pred.max().item(), y.max().item())) + 1

    cm = ConfusionMatrix(num_classes=num_classes, device=y_pred.device)
    y_pred_oh = F.one_hot(y_pred.long(), num_classes).float()
    cm.update((y_pred_oh, y.long()))
    conf = cm.compute().double()

    return _kappa_from_conf(conf, weights)


class CohenKappa(EpochMetric):
    """Compute different types of Cohen's Kappa: Non-Weighted, Linear, Quadratic.
    Buffers predictions and targets across batches, then computes the confusion
    matrix via :class:`~ignite.metrics.ConfusionMatrix` and applies the formula:
    κ = (p_o - p_e) / (1 - p_e), where p_o is the observed agreement and p_e
    is the expected agreement computed from marginal probabilities.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        weights: a string is used to define the type of Cohen's Kappa whether Non-Weighted or Linear
            or Quadratic. Default, None.
        device: optional device specification for internal storage.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_transform`` can be added
        to the metric to transform the output into the form expected by the metric.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = CohenKappa()
            metric.attach(default_evaluator, 'ck')
            y_true = torch.tensor([2, 0, 2, 2, 0, 1])
            y_pred = torch.tensor([0, 0, 2, 2, 0, 2])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['ck'])

        .. testoutput::

            0.4285...

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.

    .. versionchanged:: 0.6.0
        Replaced scikit-learn dependency with a native PyTorch implementation.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        weights: Literal["linear", "quadratic"] | None = None,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if weights not in (None, "linear", "quadratic"):
            raise ValueError("Kappa Weighting type must be None or linear or quadratic.")

        self.weights: Literal["linear", "quadratic"] | None = weights

        super().__init__(
            compute_fn=partial(_cohen_kappa_score, weights=weights),
            output_transform=output_transform,
            check_compute_fn=False,
            device=device,
            skip_unrolling=skip_unrolling,
        )

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        if y_pred.ndim > 1 or y.ndim > 1:
            raise ValueError("multilabel-indicator is not supported")

        super().update((y_pred, y))

    def compute(self) -> float:
        try:
            return super().compute()
        except NotComputableError:
            raise NotComputableError("CohenKappa must have at least one example before it can be computed.")
