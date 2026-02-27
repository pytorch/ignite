import torch

from ignite.metrics import Metric
from ignite.metrics.precision import _BasePrecisionRecall
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

__all__ = ["TopK"]


class TopK(Metric):
    def __init__(self, base_metric: Metric, top_k: int | list[int], **kwargs):
        if not hasattr(base_metric, "_prepare_output"):
            raise ValueError(f"{type(base_metric).__name__} does not support TopK.")

        self._base_metric = base_metric
        self._ks = sorted(top_k) if isinstance(top_k, list) else [top_k]
        self._states = {k: {} for k in self._ks}
        super(TopK, self).__init__(**kwargs)

    def _wrap_prepare_output(self, output, k):
        if isinstance(self._base_metric, _BasePrecisionRecall):
            y_pred, y = output[0], output[1]
            _, top_indices = torch.topk(y_pred, k=k, dim=-1)
            masked = torch.zeros_like(y_pred)
            masked.scatter_(-1, top_indices, 1.0)
            return (masked, y)

        raise ValueError(f"{type(self._base_metric).__name__} is not supported by TopK.")

    @reinit__is_reduced
    def reset(self):
        self._states = {k: {} for k in self._ks}
        self._base_metric.reset()

    @reinit__is_reduced
    def update(self, output):
        self._base_metric._check_shape(output)
        self._base_metric._check_type(output)
        self._base_metric._skip_checks = True

        for k in self._ks:
            # restore state for this k
            for attr in self._base_metric._state_dict_all_req_keys:
                setattr(self._base_metric, attr, self._states[k].get(attr, getattr(self._base_metric, attr)))

            modified_output = self._wrap_prepare_output(output, k)
            self._base_metric.update(modified_output)

            # save state for this k
            for attr in self._base_metric._state_dict_all_req_keys:
                self._states[k][attr] = getattr(self._base_metric, attr)

        self._base_metric._skip_checks = False

    @sync_all_reduce
    def compute(self) -> list:
        results = []
        for k in self._ks:
            for attr in self._base_metric._state_dict_all_req_keys:
                setattr(self._base_metric, attr, self._states[k][attr])
            results.append(self._base_metric.compute())
        return results
