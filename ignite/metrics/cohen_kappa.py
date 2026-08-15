from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics.epoch_metric import EpochMetric
from ignite.metrics.metric import Metric, reinit__is_reduced


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
            w = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)).to(dtype=conf.dtype)
        else:
            w = ((idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2).to(dtype=conf.dtype)

        w = w / w.max()
        p_o = 1 - (w * conf).sum() / n
        row = conf.sum(dim=1)
        col = conf.sum(dim=0)
        expected = row.unsqueeze(1) * col.unsqueeze(0) / n
        p_e = 1 - (w * expected).sum() / n

    epsilon = 1e-9
    return ((p_o - p_e) / (1 - p_e).clamp(min=epsilon)).item()


def _cohen_kappa_score(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    weights: Literal["linear", "quadratic"] | None,
    double_dtype: torch.dtype,
) -> float:
    if y_pred.ndim > 1 or y.ndim > 1:
        raise ValueError("multilabel-indicator is not supported")

    num_classes = int(max(y_pred.max().item(), y.max().item())) + 1

    # Build the confusion matrix locally with plain tensor ops. We must NOT use the
    # ``ConfusionMatrix`` metric here: its ``compute()`` is wrapped with ``sync_all_reduce``
    # and this function runs on rank 0 only (see ``EpochMetric.compute``), so a collective
    # would deadlock the other ranks.
    y, y_pred = y.long(), y_pred.long()
    indices = num_classes * y + y_pred
    conf = torch.bincount(indices, minlength=num_classes**2)
    conf = conf.reshape(num_classes, num_classes).to(dtype=double_dtype)

    return _kappa_from_conf(conf, weights)


class _CohenKappaEpochMetric(EpochMetric):
    """CohenKappa backed by EpochMetric — infers num_classes dynamically from data."""

    def __init__(
        self,
        weights: Literal["linear", "quadratic"] | None,
        output_transform: Callable,
        device: str | torch.device,
        skip_unrolling: bool,
        check_compute_fn: bool,
    ):
        super().__init__(
            # ``self._double_dtype`` (set in Metric.__init__, float32 on MPS / float64 otherwise)
            # is resolved lazily at compute time.
            compute_fn=lambda y_pred, y: _cohen_kappa_score(y_pred, y, weights, self._double_dtype),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
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

        super().update((y_pred, y))


class _CohenKappaConfusionMatrix(Metric):
    """CohenKappa backed by ConfusionMatrix — requires num_classes at construction time.
    Accumulates a running confusion matrix; no raw tensor buffering.
    """

    _state_dict_all_req_keys = ("_cm",)

    def __init__(
        self,
        num_classes: int,
        weights: Literal["linear", "quadratic"] | None,
        output_transform: Callable,
        device: str | torch.device,
        skip_unrolling: bool,
    ):
        self._weights = weights
        self._cm = ConfusionMatrix(
            num_classes=num_classes,
            output_transform=output_transform,
            device=device,
            skip_unrolling=skip_unrolling,
        )
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._cm.reset()

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        if y_pred.ndim > 1 or y.ndim > 1:
            raise ValueError("multilabel-indicator is not supported")

        num_classes = self._cm.num_classes
        y_pred_oh = F.one_hot(y_pred.long(), num_classes).float().to(self._device)
        self._cm.update((y_pred_oh, y.long().to(self._device)))

    def compute(self) -> float:
        conf = self._cm.compute().to(dtype=self._double_dtype)
        return _kappa_from_conf(conf, self._weights)


class CohenKappa(Metric):
    """Compute different types of Cohen's Kappa: Non-Weighted, Linear, Quadratic.

    When ``num_classes`` is provided, accumulates a running confusion matrix via
    :class:`~ignite.metrics.confusion_matrix.ConfusionMatrix` (memory-efficient, no raw tensor buffering).
    When ``num_classes`` is ``None`` (default), buffers predictions and targets via
    :class:`~ignite.metrics.EpochMetric` and infers the number of classes from data.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        weights: a string is used to define the type of Cohen's Kappa whether Non-Weighted or Linear
            or Quadratic. Default, None.
        check_compute_fn: Default False. If True, the compute function is run on the first batch
            of data to ensure there are no issues. User will be warned in case there are any issues
            computing the function.
        device: optional device specification for internal storage.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.
        num_classes: number of classes. If provided, uses a running confusion matrix
            (memory-efficient). If ``None``, infers from data at compute time (backward-compatible default).

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

    .. versionchanged:: 0.5.5
        Replaced scikit-learn dependency with a native PyTorch implementation.
        Added ``num_classes`` argument; routes to a running-confusion-matrix backend when provided.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        weights: Literal["linear", "quadratic"] | None = None,
        check_compute_fn: bool = False,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
        num_classes: int | None = None,
    ):
        if weights not in (None, "linear", "quadratic"):
            raise ValueError("Kappa Weighting type must be None or linear or quadratic.")

        self.weights: Literal["linear", "quadratic"] | None = weights

        if num_classes is not None:
            self._impl: Metric = _CohenKappaConfusionMatrix(
                num_classes=num_classes,
                weights=weights,
                output_transform=output_transform,
                device=device,
                skip_unrolling=skip_unrolling,
            )
        else:
            self._impl = _CohenKappaEpochMetric(
                weights=weights,
                output_transform=output_transform,
                device=device,
                skip_unrolling=skip_unrolling,
                check_compute_fn=check_compute_fn,
            )

        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._impl.reset()

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._impl.update(output)

    def compute(self) -> float:
        return self._impl.compute()
