from typing import Any, Callable, Dict, Sequence, Tuple

import torch

from ignite.metrics import Metric


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
            true for multi-output model, for example, if ``y_pred`` and ``y`` contain multi-ouput as
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

    _state_dict_all_req_keys: Tuple[str, ...] = ("metrics",)

    def __init__(
        self, metrics: Dict[str, Metric], output_transform: Callable = lambda x: x, skip_unrolling: bool = False
    ):
        self.metrics = metrics
        super(MetricGroup, self).__init__(output_transform=output_transform, skip_unrolling=skip_unrolling)

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def update(self, output: Sequence[torch.Tensor]) -> None:
        for m in self.metrics.values():
            m.update(m._output_transform(output))

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.metrics.items()}
