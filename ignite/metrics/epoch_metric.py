import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Union, cast

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced

# Supported return types for ``EpochMetric``'s ``compute_fn``.
EpochMetricOutput = Union[int, float, torch.Tensor, Sequence, Mapping]

__all__ = ["EpochMetric"]

# Type tags used by ``EpochMetric._broadcast_result`` to let every rank agree on how to
# decode the next value(s) coming out of the collective calls. ``_TAG_UNSUPPORTED`` is also
# reused to reject non-str mapping keys, since both cases mean "every rank should raise".
_TAG_INT = 0
_TAG_FLOAT = 1
_TAG_TENSOR = 2
_TAG_TUPLE = 3
_TAG_LIST = 4
_TAG_MAPPING = 5
_TAG_UNSUPPORTED = -1


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

    - ``update`` must receive output of the form ``(y_pred, y)``.

    Args:
        compute_fn: a callable which receives two tensors as the `predictions` and `targets`
            and returns the computed metric. Supported return types are: ``int``, ``float``,
            ``torch.Tensor``, a ``Sequence`` (tuple/list) of these, or a ``Mapping`` (dict) with
            string keys and these values. An unsupported return type raises a ``TypeError``.
            Note: in distributed configuration (``world_size > 1``), only scalar and
            ``torch.Tensor`` outputs are broadcast across processes; tuple/list/mapping outputs
            are supported only when ``world_size == 1``. Input tensors will be on specified
            ``device`` (see arg below).
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: if True, ``compute_fn`` is run on the first batch of data to ensure there are no
            issues. If issues exist, user is warned that there might be an issue with the ``compute_fn``.
            Default, True.
        device: optional device specification for internal storage.

    Example:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            def mse_fn(y_preds, y_targets):
                return torch.mean(((y_preds - y_targets.type_as(y_preds)) ** 2)).item()

            metric = EpochMetric(mse_fn)
            metric.attach(default_evaluator, "mse")
            y_true = torch.tensor([0, 1, 2, 3, 4, 5])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["mse"])

        .. testoutput::

            0.5729...

    Warnings:
        EpochMetricWarning: User is warned that there are issues with ``compute_fn`` on a batch of data processed.
        To disable the warning, set ``check_compute_fn=False``.

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_predictions", "_targets")

    def __init__(
        self,
        compute_fn: Callable[[torch.Tensor, torch.Tensor], float],
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = True,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        self.compute_fn = compute_fn
        self._check_compute_fn = check_compute_fn

        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._result: EpochMetricOutput | None = None

    def _check_shape(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_targets) or (batch_size, ).")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_targets) or (batch_size, ).")

    def _check_type(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
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
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
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

    def _check_output_type(self, result: EpochMetricOutput) -> None:
        # Recursively validate that compute_fn's output is a supported type. ``str``/``bytes``
        # are rejected explicitly since ``str`` is itself a ``Sequence``.
        if isinstance(result, (int, float, torch.Tensor)):
            return
        if isinstance(result, Mapping):
            for key, value in result.items():
                if not isinstance(key, str):
                    raise TypeError(f"compute_fn output mapping keys should be str, but given {type(key)}.")
                self._check_output_type(value)
            return
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            for value in result:
                self._check_output_type(value)
            return
        raise TypeError(
            f"compute_fn output type {type(result)} is not supported. Supported types are: "
            "int, float, torch.Tensor, a Sequence of these, or a Mapping with str keys and these values."
        )

    def _broadcast_result(self, result: EpochMetricOutput, src: int = 0) -> EpochMetricOutput:
        """Recursively broadcast compute_fn output from src rank to all ranks.

        Each step only broadcasts types that ``idist.broadcast`` natively supports
        (int, float, torch.Tensor, str), so containers are transmitted by first
        synchronising their structure (type tag, length, dict keys) and then
        broadcasting each leaf individually.

        Every rank must take the same path through the collective calls below, so any
        rejection (unsupported type, non-str mapping key) is decided from a value that has
        already been broadcast to all ranks, never from a check that only src has run. This
        way every rank raises together instead of some ranks hanging on a broadcast that src
        never issues.
        """
        rank = idist.get_rank()

        # Step 1: broadcast type tag so all ranks know what to expect
        if rank == src:
            if isinstance(result, int):
                tag = _TAG_INT
            elif isinstance(result, float):
                tag = _TAG_FLOAT
            elif isinstance(result, torch.Tensor):
                tag = _TAG_TENSOR
            elif isinstance(result, tuple):
                tag = _TAG_TUPLE
            elif isinstance(result, list):
                tag = _TAG_LIST
            elif isinstance(result, Mapping):
                tag = _TAG_MAPPING
            else:
                tag = _TAG_UNSUPPORTED
        else:
            tag = _TAG_INT
        tag = int(idist.broadcast(tag, src=src))

        if tag == _TAG_UNSUPPORTED:
            raise TypeError(
                f"compute_fn output type is not supported. Supported types are: "
                "int, float, torch.Tensor, a Sequence of these, or a Mapping with str keys and these values."
            )

        # Step 2: broadcast content based on type
        if tag == _TAG_INT:
            return int(idist.broadcast(result if rank == src else 0, src=src))
        if tag == _TAG_FLOAT:
            return float(idist.broadcast(result if rank == src else 0.0, src=src))
        if tag == _TAG_TENSOR:
            return cast(torch.Tensor, idist.broadcast(result if rank == src else None, src=src, safe_mode=True))

        if tag in (_TAG_TUPLE, _TAG_LIST):
            length = int(idist.broadcast(len(result) if rank == src else 0, src=src))
            elements = []
            for i in range(length):
                elem = result[i] if rank == src else None
                elements.append(self._broadcast_result(elem, src=src))
            return tuple(elements) if tag == _TAG_TUPLE else elements

        # tag == _TAG_MAPPING
        src_keys = list(result.keys()) if rank == src else []
        n_keys = int(idist.broadcast(len(src_keys) if rank == src else 0, src=src))
        keys = []
        for i in range(n_keys):
            raw_key = src_keys[i] if rank == src else None
            # Broadcast whether this key is a valid (str) key before broadcasting the key
            # itself: src and non-src ranks must agree on the wire type (str) they are about
            # to exchange, so this can't be decided from a src-only isinstance check.
            key_valid = int(idist.broadcast(0 if (rank != src or isinstance(raw_key, str)) else -1, src=src))
            if key_valid == _TAG_UNSUPPORTED:
                detail = f" but given {type(raw_key)}" if rank == src else ""
                raise TypeError(f"compute_fn output mapping keys should be str{detail}.")
            key = str(idist.broadcast(raw_key if rank == src else "", src=src))
            keys.append(key)
        values = []
        for i in range(n_keys):
            val = result[keys[i]] if rank == src else None
            values.append(self._broadcast_result(val, src=src))
        return dict(zip(keys, values))

    def compute(self) -> EpochMetricOutput:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(f"{type(self).__name__} must have at least one example before it can be computed.")

        if self._result is None:
            _prediction_tensor = torch.cat(self._predictions, dim=0)
            _target_tensor = torch.cat(self._targets, dim=0)

            ws = idist.get_world_size()
            if ws > 1:
                # All gather across all processes
                _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
                _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))

            result: EpochMetricOutput = 0.0
            if idist.get_rank() == 0:
                # Run compute_fn on zero rank only
                result = self.compute_fn(_prediction_tensor, _target_tensor)

            if ws > 1:
                # Type/key validation happens inside `_broadcast_result` itself (via the tag
                # protocol), so every rank reaches the same TypeError together. Do not
                # pre-validate on rank 0 alone here: that would let rank 0 raise and return
                # before issuing the first broadcast, leaving other ranks waiting on a
                # collective call rank 0 never makes.
                result = self._broadcast_result(result, src=0)
            else:
                self._check_output_type(result)

            self._result = result

        return self._result


class EpochMetricWarning(UserWarning):
    pass
