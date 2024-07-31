from typing import Any, Callable, Dict, Sequence

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
    """

    _state_dict_all_req_keys = ("metrics",)

    def __init__(self, metrics: Dict[str, Metric], output_transform: Callable = lambda x: x):
        self.metrics = metrics
        super(MetricGroup, self).__init__(output_transform=output_transform)

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    def update(self, output: Sequence[torch.Tensor]) -> None:
        for m in self.metrics.values():
            m.update(m._output_transform(output))

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.metrics.items()}
