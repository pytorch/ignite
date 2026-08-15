import copy
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced

__all__ = ["SubgroupMetric", "SubgroupDifference"]


class _SubgroupBase(Metric):
    """Internal base class for metrics that compute results per subgroup."""

    required_output_keys = ("y_pred", "y", "group_labels")

    def __init__(
        self,
        base_metric: Metric,
        groups: Sequence[Any],
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
    ) -> None:
        self._base_metric = base_metric
        self._user_groups = list(groups)
        self._metrics: dict[Any, Metric] = {g: copy.deepcopy(base_metric) for g in self._user_groups}
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y, group_labels = output[0].detach(), output[1].detach(), output[2].detach()

        if y_pred.shape[0] != y.shape[0] or y_pred.shape[0] != group_labels.shape[0]:
            raise ValueError("y_pred, y, and group_labels must have the same batch size.")

        for g, metric in self._metrics.items():
            mask = group_labels == g
            if mask.any():
                metric.update((y_pred[mask], y[mask]))

    def _compute_groups(self) -> dict[Any, Any]:
        results = {}
        for g, metric in self._metrics.items():
            try:
                results[g] = metric.compute()
            except NotComputableError:
                continue
        return results

    def state_dict(self) -> OrderedDict:
        state_dict = super().state_dict()
        state_dict["_metrics"] = {str(g): m.state_dict() for g, m in self._metrics.items()}
        return state_dict

    def load_state_dict(self, state_dict: Mapping) -> None:
        state_dict_dict = dict(state_dict)
        metrics_state = state_dict_dict.pop("_metrics")
        super().load_state_dict(state_dict_dict)
        for g, m_state in metrics_state.items():
            found_g = None
            for original_g in self._metrics.keys():
                if str(original_g) == g:
                    found_g = original_g
                    break
            if found_g is not None:
                self._metrics[found_g].load_state_dict(m_state)


class SubgroupMetric(_SubgroupBase):
    """A wrapper metric that computes a base metric for each unique subgroup in the dataset.

    This class handles slicing the input data (y_pred, y) according to group labels and
    maintains independent state for each group by delegating to a dictionary of metrics.

    Args:
        base_metric: an instance of the metric to be computed for each group.
            This metric will be cloned for each group.
        groups: a sequence of unique group identifiers.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. Default is ``(y_pred, y, group_labels)``.
        device: specifies which device updates are accumulated on.

    .. versionadded:: 0.5.4
    """

    def compute(self) -> dict[Any, Any]:
        """Computes the metric for each subgroup.

        Returns:
            A dictionary mapping group identifiers to their respective metric results.
        """
        return self._compute_groups()


class SubgroupDifference(_SubgroupBase):
    """A wrapper metric that computes the maximum difference between any two subgroups' metric values.

    This metric is useful for measuring performance disparities across demographic segments.

    Args:
        base_metric: an instance of the metric to be computed for each group.
        groups: a sequence of unique group identifiers.
        output_transform: a callable that is used to transform the engine output.
        device: specifies the computation device.

    .. versionadded:: 0.5.4
    """

    def compute(self) -> float:
        """Computes the maximum disparity between any two subgroups.

        Returns:
            The maximum difference (max - min) in metric value across all non-empty subgroups.

        Raises:
            NotComputableError: if less than two unique subgroups have been processed.
        """
        group_results = self._compute_groups()

        if len(group_results) == 0:
            raise NotComputableError("Fairness metrics must have at least one example before it can be computed.")

        if len(group_results) < 2:
            raise NotComputableError("Fairness metrics require at least two unique subgroups to compute a disparity.")

        values = list(group_results.values())
        if all(isinstance(v, torch.Tensor) for v in values):
            stacked = torch.stack(values)
            # Compute disparity per-index/per-class, then take the maximum disparity
            disparities = stacked.max(dim=0).values - stacked.min(dim=0).values
            return float(disparities.max().item())

        return float(max(values) - min(values))
