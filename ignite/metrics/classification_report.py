from typing import Callable, Dict, List, Optional, Sequence, Union

import torch

from ignite.metrics.metric import reinit__is_reduced
from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["ClassificationReport"]


class ClassificationReport(MetricsLambda):
    def __init__(
        self,
        output_dict: bool = False,
        output_transform: Optional[Callable] = None,
        device: Union[str, torch.device] = torch.device("cpu"),
        labels: Optional[List[str]] = None,
    ):
        self.precision = Precision(average=False, device=device,)
        self.recall = Recall(average=False, device=device,)
        self.averaged_precision = Precision(average=True, device=device,)
        self.averaged_recall = Recall(average=True, device=device,)
        self.output_dict = output_dict
        self.output_transform = output_transform
        self.labels = labels
        self.combined_metric = MetricsLambda(
            self._wrapper, self.recall, self.precision, self.averaged_recall, self.averaged_precision,
        )

    def _wrapper(self, r, p, a_r, a_p):
        p_tensor, r_tensor = p, r
        assert p_tensor.shape == r_tensor.shape
        dict_obj = {}
        for idx, p_label in enumerate(p_tensor):
            dict_obj[self._get_label_for_class(idx)] = {
                "precision": p_label.item(),
                "recall": r_tensor[idx].item(),
            }
        dict_obj["macro_avg"] = {
            "precision": a_r,
            "recall": a_p,
        }
        result = dict_obj if self.output_dict else str(dict_obj)
        return self.output_transform(result) if self.output_transform else result

    def _get_label_for_class(self, idx: int) -> str:
        return self.labels[idx] if self.labels else str(idx)

    @reinit__is_reduced
    def reset(self) -> None:
        self.combined_metric.reset()

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        self.precision.update(output)
        self.averaged_precision.update(output)
        self.recall.update(output)
        self.averaged_recall.update(output)

    def compute(self) -> Union[Dict, str]:
        return self.combined_metric.compute()
