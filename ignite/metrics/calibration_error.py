from collections.abc import Callable, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["ExpectedCalibrationError", "MaximumCalibrationError"]


class _BaseCalibrationError(Metric):
    """Base class accumulating the per-bin statistics shared by calibration metrics.

    Predictions are grouped into ``num_bins`` equal-width confidence bins. For each bin only three
    fixed-size aggregates are kept: the number of correct predictions, the sum of confidences and the
    number of samples. This makes the memory cost ``O(num_bins)`` (independent of the number of
    samples) and lets the final result be computed without any Python loop over bins.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be **probabilities** (already normalized with softmax/sigmoid). For
      multiclass inputs :math:`(B, C)` or :math:`(B, C, ...)` are allowed and the confidence of a
      sample is its maximum class probability. For binary inputs :math:`(B,)` or :math:`(B, 1)`,
      ``y_pred`` is the probability of the positive class and the confidence is
      :math:`\\max(p, 1 - p)`. If your model outputs logits, apply an activation through
      ``output_transform``.
    - ``y`` holds the ground truth class indices (or 0/1 labels in the binary case).
    """

    _state_dict_all_req_keys = ("_bin_correct", "_bin_conf", "_bin_count")

    def __init__(
        self,
        num_bins: int = 10,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if not isinstance(num_bins, int) or num_bins < 1:
            raise ValueError(f"Argument num_bins must be a positive integer, got {num_bins}.")
        self.num_bins = num_bins
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._bin_correct = torch.zeros(self.num_bins, device=self._device)
        self._bin_conf = torch.zeros(self.num_bins, device=self._device)
        self._bin_count = torch.zeros(self.num_bins, device=self._device)

    def _confidence_and_correct(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y_pred.ndim == y.ndim + 1:
            num_classes = y_pred.shape[1]
            if num_classes >= 2:
                # (B, C, ...) -> (B, ..., C) -> (B*..., C), regarding as B*... predictions
                y_pred = y_pred.movedim(1, -1).reshape(-1, num_classes)
                conf, pred = y_pred.max(dim=1)
                y = y.reshape(-1)
            else:
                # (B, 1, ...) single positive-class probability -> binary
                y_pred = y_pred.reshape(-1)
                y = y.reshape(-1)
                conf = torch.maximum(y_pred, 1.0 - y_pred)
                pred = (y_pred >= 0.5).long()
        elif y_pred.ndim == y.ndim:
            # binary: y_pred is the probability of the positive class
            y_pred = y_pred.reshape(-1)
            y = y.reshape(-1)
            conf = torch.maximum(y_pred, 1.0 - y_pred)
            pred = (y_pred >= 0.5).long()
        else:
            raise ValueError(
                "y_pred must have shape (B, C) or (B, C, ...) for multiclass inputs or (B,) or (B, 1) "
                "for binary inputs, and be compatible with y; "
                f"got y_pred shape {tuple(y_pred.shape)} and y shape {tuple(y.shape)}."
            )

        correct = pred.eq(y.view_as(pred)).to(self._device, dtype=self._bin_count.dtype)
        conf = conf.to(self._device, dtype=self._bin_count.dtype)
        return conf, correct

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        conf, correct = self._confidence_and_correct(y_pred, y)

        bin_idx = torch.clamp((conf * self.num_bins).long(), 0, self.num_bins - 1)
        self._bin_count.index_add_(0, bin_idx, torch.ones_like(conf))
        self._bin_conf.index_add_(0, bin_idx, conf)
        self._bin_correct.index_add_(0, bin_idx, correct)

    def _accuracy_confidence_gap(self) -> torch.Tensor:
        nonempty = self._bin_count > 0
        acc = torch.zeros_like(self._bin_count)
        conf = torch.zeros_like(self._bin_count)
        acc[nonempty] = self._bin_correct[nonempty] / self._bin_count[nonempty]
        conf[nonempty] = self._bin_conf[nonempty] / self._bin_count[nonempty]
        return (acc - conf).abs()


class ExpectedCalibrationError(_BaseCalibrationError):
    r"""Calculates the `Expected Calibration Error (ECE)
    <https://arxiv.org/abs/1706.04599>`_.

    Predictions are grouped into :math:`M` equal-width confidence bins and the ECE is the weighted
    average of the gap between accuracy and confidence within each bin:

    .. math:: \text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|

    where :math:`B_m` is the set of samples whose confidence falls into the :math:`m`-th bin,
    :math:`n` is the total number of samples, :math:`\text{acc}(B_m)` is the accuracy within the bin
    and :math:`\text{conf}(B_m)` is the mean confidence within the bin. A perfectly calibrated model
    has :math:`\text{ECE} = 0`.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be **probabilities** (already normalized). Multiclass shapes
      :math:`(B, C)` / :math:`(B, C, ...)` and binary shapes :math:`(B,)` / :math:`(B, 1)` are
      supported. If your model outputs logits, apply an activation through ``output_transform``.

    Args:
        num_bins: number of equal-width confidence bins used to group the predictions. Default 10.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = ExpectedCalibrationError(num_bins=5)
            metric.attach(default_evaluator, 'ece')
            y_true = torch.tensor([0, 0, 1, 0])
            y_pred = torch.tensor([
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.3, 0.7],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['ece'])

        .. testoutput::

            0.2500000...

    .. versionadded:: 0.6.0
    """

    @sync_all_reduce("_bin_correct", "_bin_conf", "_bin_count")
    def compute(self) -> float:
        n = self._bin_count.sum()
        if n == 0:
            raise NotComputableError(
                "ExpectedCalibrationError must have at least one example before it can be computed."
            )
        gap = self._accuracy_confidence_gap()
        return ((self._bin_count / n) * gap).sum().item()


class MaximumCalibrationError(_BaseCalibrationError):
    r"""Calculates the `Maximum Calibration Error (MCE)
    <https://arxiv.org/abs/1706.04599>`_.

    Predictions are grouped into :math:`M` equal-width confidence bins and the MCE is the largest gap
    between accuracy and confidence across the (non-empty) bins:

    .. math:: \text{MCE} = \max_{m \in \{1, ..., M\}} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|

    where :math:`B_m` is the set of samples whose confidence falls into the :math:`m`-th bin,
    :math:`\text{acc}(B_m)` is the accuracy within the bin and :math:`\text{conf}(B_m)` is the mean
    confidence within the bin. Unlike :class:`~ignite.metrics.ExpectedCalibrationError`, the MCE is not
    weighted by bin size and so reflects the worst-case miscalibration.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be **probabilities** (already normalized). Multiclass shapes
      :math:`(B, C)` / :math:`(B, C, ...)` and binary shapes :math:`(B,)` / :math:`(B, 1)` are
      supported. If your model outputs logits, apply an activation through ``output_transform``.

    Args:
        num_bins: number of equal-width confidence bins used to group the predictions. Default 10.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-output as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MaximumCalibrationError(num_bins=5)
            metric.attach(default_evaluator, 'mce')
            y_true = torch.tensor([0, 0, 1, 0])
            y_pred = torch.tensor([
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.3, 0.7],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['mce'])

        .. testoutput::

            0.6999999...

    .. versionadded:: 0.6.0
    """

    @sync_all_reduce("_bin_correct", "_bin_conf", "_bin_count")
    def compute(self) -> float:
        if self._bin_count.sum() == 0:
            raise NotComputableError(
                "MaximumCalibrationError must have at least one example before it can be computed."
            )
        nonempty = self._bin_count > 0
        return self._accuracy_confidence_gap()[nonempty].max().item()
