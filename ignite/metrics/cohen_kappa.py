from collections.abc import Callable
from typing import Literal

import torch

from ignite.metrics.epoch_metric import EpochMetric


class CohenKappa(EpochMetric):
    """Compute different types of Cohen's Kappa: Non-Wieghted, Linear, Quadratic.
    Accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        weights: a string is used to define the type of Cohen's Kappa whether Non-Weighted or Linear
            or Quadratic. Default, None.
        check_compute_fn: Default False. If True, `cohen_kappa_score
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>`_
            is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
        device: optional device specification for internal storage.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
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
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        weights: Literal["linear", "quadratic"] | None = None,
        check_compute_fn: bool = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if weights not in (None, "linear", "quadratic"):
            raise ValueError("Kappa Weighting type must be None or linear or quadratic.")

        # initialize weights
        self.weights: Literal["linear", "quadratic"] | None = weights

        super().__init__(
            self._cohen_kappa_score,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
            skip_unrolling=skip_unrolling,
        )

    def _cohen_kappa_score(self, y_targets: torch.Tensor, y_preds: torch.Tensor) -> float:
        if y_targets.ndim > 1 or y_preds.ndim > 1:
            raise ValueError("multilabel-indicator is not supported")
        n_classes = int(max(y_targets.max().item(), y_preds.max().item())) + 1

        indices = y_targets * n_classes + y_preds
        conf = torch.bincount(indices, minlength=n_classes * n_classes).reshape(n_classes, n_classes).double()
        n = conf.sum()

        if self.weights is None:
            p_o = conf.trace() / n
            row = conf.sum(dim=1)
            col = conf.sum(dim=0)
            p_e = (row * col).sum() / (n * n)

        else:
            idx = torch.arange(n_classes, device=y_targets.device)
            if self.weights == "linear":
                w = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).double()
            else:
                w = ((idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2).double()

            w = w / w.max()
            p_o = 1 - (w * conf).sum() / n
            row = conf.sum(dim=1)
            col = conf.sum(dim=0)
            expected = row.unsqueeze(1) * col.unsqueeze(0) / n
            p_e = 1 - (w * expected).sum() / n

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa.item()
