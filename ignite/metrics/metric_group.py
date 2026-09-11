from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import _is_list_of_tensors_or_numbers, _to_batched_tensor


class MetricGroup(Metric):
    """
    A class for grouping metrics so that user could manage them easier.

    Args:
        metrics: a dictionary of names to metric instances.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. `output_transform` of each metric in the group is also
            called upon its update.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` and ``y`` contain multi-output as
            ``(y_pred_a, y_pred_b)`` and ``(y_a, y_b)``, in which case the update method is called for
            ``(y_pred_a, y_a)`` and ``(y_pred_b, y_b)``.Alternatively, ``output_transform`` can be used to handle
            this.

    Examples:
        We construct a group of metrics, attach them to the engine at once and retrieve their result.

        .. code-block:: python

           import torch

           metric_group = MetricGroup({'acc': Accuracy(), 'precision': Precision(), 'loss': Loss(nn.NLLLoss())})
           metric_group.attach(default_evaluator, "eval_metrics")
           y_true = torch.tensor([1, 0, 1, 1, 0, 1])
           y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
           state = default_evaluator.run([[y_pred, y_true]])

           # Metrics individually available in `state.metrics`
           state.metrics["acc"], state.metrics["precision"], state.metrics["loss"]

           # And also altogether
           state.metrics["eval_metrics"]

    .. versionchanged:: 0.5.2
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys: tuple[str, ...] = ("metrics",)

    def __init__(
        self, metrics: dict[str, Metric], output_transform: Callable = lambda x: x, skip_unrolling: bool = False
    ):
        self.metrics = metrics
        super().__init__(output_transform=output_transform, skip_unrolling=skip_unrolling)

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def iteration_completed(self, engine: Engine) -> None:
        # Overridden because, unlike a "leaf" metric, a MetricGroup does not itself consume a
        # ``(y_pred, y)``-shaped output: each metric in the group applies its own
        # ``output_transform`` in ``update`` to pull whatever it needs out of the group's
        # (transformed) output. So, unlike ``Metric.iteration_completed``, a mapping output is
        # passed straight through to ``update`` rather than being validated/unpacked against
        # ``required_output_keys``, which only makes sense for a single metric's ``(y_pred, y)``.
        output = self._output_transform(engine.state.output)
        if isinstance(output, Mapping):
            self.update(output)
            return

        if (
            (not self._skip_unrolling)
            and isinstance(output, Sequence)
            and all(_is_list_of_tensors_or_numbers(o) for o in output)
        ):
            if not (len(output) == 2 and len(output[0]) == len(output[1])):
                raise ValueError(
                    f"Output should have 2 items of the same length, "
                    f"got {len(output)} and {len(output[0])}, {len(output[1])}"
                )
            for o1, o2 in zip(output[0], output[1]):
                # o1 and o2 are list of tensors or numbers
                tensor_o1 = _to_batched_tensor(o1)
                tensor_o2 = _to_batched_tensor(o2, device=tensor_o1.device)
                self.update((tensor_o1, tensor_o2))
        else:
            self.update(output)

    def update(self, output: Sequence[torch.Tensor] | Mapping[Any, Any]) -> None:
        for m in self.metrics.values():
            m.update(m._output_transform(output))

    def compute(self) -> dict[str, Any]:
        return {k: m.compute() for k, m in self.metrics.items()}
