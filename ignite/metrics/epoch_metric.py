import warnings
from typing import Callable, List, Tuple, Union, cast

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced

__all__ = ["EpochMetric"]


class EpochMetric(Metric):
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape ``(batch_size, n_targets)``. Output
    datatype should be `float32`. Target datatype should be `long` for classification and `float` for regression.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

        In distributed configuration, all stored data (output and target) is mutually collected across all processes
        using all gather collective operation. This can potentially lead to a memory error.
        Compute method executes ``compute_fn`` on zero rank process only and final result is broadcasted to
        all processes.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.

    Args:
        compute_fn: a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar. Input tensors will be on specified ``device``
            (see arg below).
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: if True, ``compute_fn`` is run on the first batch of data to ensure there are no
            issues. If issues exist, user is warned that there might be an issue with the ``compute_fn``.
            Default, True.
        device: optional device specification for internal storage.

    Example:

        .. testcode::

            def mse_fn(y_preds, y_targets):
                return torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2)).item()

            metric = EpochMetric(mse_fn)
            metric.attach(default_evaluator, "mse")
            y_true = torch.Tensor([0, 1, 2, 3, 4, 5])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["mse"])

        .. testoutput::

            0.5729...

    Warnings:
        EpochMetricWarning: User is warned that there are issues with ``compute_fn`` on a batch of data processed.
        To disable the warning, set ``check_compute_fn=False``.
    """

    def __init__(
        self,
        compute_fn: Callable,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        self.compute_fn = compute_fn
        self._check_compute_fn = check_compute_fn

        super(EpochMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._predictions = []  # type: List[torch.Tensor]
        self._targets = []  # type: List[torch.Tensor]

    def _check_shape(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_targets) or (batch_size, ).")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_targets) or (batch_size, ).")

    def _check_type(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        if len(self._predictions) < 1:
            return
        dtype_preds = self._predictions[-1].dtype
        if dtype_preds != y_pred.dtype:
            raise ValueError(
                f"Incoherent types between input y_pred and stored predictions: {dtype_preds} vs {y_pred.dtype}"
            )

        dtype_targets = self._targets[-1].dtype
        if dtype_targets != y.dtype:
            raise ValueError(f"Incoherent types between input y and stored targets: {dtype_targets} vs {y.dtype}")

    @reinit__is_reduced
    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output[0].detach(), output[1].detach()

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.clone().to(self._device)
        y = y.clone().to(self._device)

        self._check_type((y_pred, y))
        self._predictions.append(y_pred)
        self._targets.append(y)

        # Check once the signature and execution of compute_fn
        if len(self._predictions) == 1 and self._check_compute_fn:
            try:
                self.compute_fn(self._predictions[0], self._targets[0])
            except Exception as e:
                warnings.warn(f"Probably, there can be a problem with `compute_fn`:\n {e}.", EpochMetricWarning)

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("EpochMetric must have at least one example before it can be computed.")

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        ws = idist.get_world_size()

        if ws > 1 and not self._is_reduced:
            # All gather across all processes
            _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
            _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))
        self._is_reduced = True

        result = 0.0
        if idist.get_rank() == 0:
            # Run compute_fn on zero rank only
            result = self.compute_fn(_prediction_tensor, _target_tensor)

        if ws > 1:
            # broadcast result to all processes
            result = cast(float, idist.broadcast(result, src=0))

        return result


class EpochMetricWarning(UserWarning):
    pass
