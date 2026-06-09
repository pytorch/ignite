import torch
from typing import Sequence

from ignite.metrics import Metric
from ignite.metrics.precision import _BasePrecisionRecall


class TopK(Metric):
    """https://github.com/open-mmlab/mmengine/blob/main/mmengine/registry/registry.py
        https://github.com/open-mmlab/mmdetection/tree/main

    the idea is to maintain top_k transforms here for metrics.
    each topk_transform will be registered in TopK class.
    and user will have to pass output_transform to TopK instead of base_metric
    and the output transform must only unpack output into y_pred, y, and do no other transformations like binarisation
    """

    _output_transform_registry = {}

    @classmethod
    def register(cls, metric_type, k_transform):
        cls._output_transform_registry[metric_type] = k_transform

    def __init__(
        self,
        base_metric: Metric,
        top_k: int | list[int],
        output_transform=lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        transform = None
        for metric_type, k_transform in self._output_transform_registry.items():
            if isinstance(base_metric, metric_type):
                transform = k_transform

        if transform is None:
            raise ValueError(f"{type(base_metric).__name__} does not support TopK.")

        self._transform = transform
        self._base_metric = base_metric
        self._ks = sorted(top_k) if isinstance(top_k, list) else [top_k]
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    def reset(self):
        self._base_metric.reset()
        self._states = {k: self._base_metric.state_dict() for k in self._ks}

    def update(self, output):
        """run checks on output only once for highest `k` value and then skip checks
        in base metric's update for each `k`"""
        _output = self._transform(output, self._ks[-1])
        self._base_metric._check_shape(_output)
        self._base_metric._check_type(_output)
        self._base_metric._skip_checks = True

        for k in self._ks:
            # restore state for this k
            self._base_metric.load_state_dict(self._states[k])

            k_output = self._transform(output, k)
            self._base_metric.update(k_output)

            # save state for this k
            self._states[k] = self._base_metric.state_dict()

        self._base_metric._skip_checks = False

    def compute(self) -> list:
        results = []
        for k in self._ks:
            self._base_metric.load_state_dict(self._states[k])
            results.append(self._base_metric.compute())
        return results


def _precision_recall_topk_transform(output: Sequence[torch.Tensor], k: int):
    """top_k transform for precision and recall"""
    y_pred, y = output[0], output[1]
    _, top_indices = torch.topk(y_pred, k=k, dim=-1)
    masked = torch.zeros_like(y_pred)
    masked.scatter_(-1, top_indices, 1.0)
    return (masked, y)


TopK.register(_BasePrecisionRecall, _precision_recall_topk_transform)
