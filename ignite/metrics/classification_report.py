import json
from typing import Callable, Collection, Dict, List, Optional, Union

import torch

from ignite.metrics.fbeta import Fbeta
from ignite.metrics.metric import Metric
from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["ClassificationReport"]


def ClassificationReport(
    beta: int = 1,
    output_dict: bool = False,
    output_transform: Callable = lambda x: x,
    device: Union[str, torch.device] = torch.device("cpu"),
    labels: Optional[List[str]] = None,
) -> MetricsLambda:
    """Build a text report showing the main classification metrics. The report resembles in functionality to
         `scikit-learn classification_report
         <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report>`_
        The underlying implementation doesn't use the sklearn function.

        Args:
            beta: weight of precision in harmonic mean
            output_dict: If True, return output as dict, otherwise return a str
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
            device: optional device specification for internal storage.
            labels: Optional list of label indices to include in the report

        .. code-block:: python


            y_true = torch.tensor([[1, 0],
            [0, 1], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]])
            y_pred = torch.randint(0, 2, size=(10,))

            classification_report = ClassificationReport(output_dict=True, digits=2)
            classification_report.update((y_true, y_pred))
            res = classification_report.compute()

            # result should look like this:
            {
                "0": {
                    "precision": 0.33,
                    "recall": 0.5,
                    "f1-score": 0.4
                },
                "1": {
                    "precision": 0.5,
                    "recall": 0.33,
                    "f1-score": 0.4
                },
                "macro avg": {
                    "precision": 0.42,
                    "recall": 0.42,
                    "f1-score": 0.4
                }
            }
    """

    # setup all the underlying metrics
    precision = Precision(average=False, output_transform=output_transform, device=device,)
    recall = Recall(average=False, output_transform=output_transform, device=device,)
    fbeta = Fbeta(beta, average=False, precision=precision, recall=recall)
    averaged_precision = precision.mean()
    averaged_recall = recall.mean()
    averaged_fbeta = fbeta.mean()

    def _wrapper(
        recall_metric: Metric, precision_metric: Metric, f: Metric, a_recall: Metric, a_precision: Metric, a_f: Metric,
    ) -> Union[Collection[str], Dict]:
        p_tensor, r_tensor, f_tensor = precision_metric, recall_metric, f
        if p_tensor.shape != r_tensor.shape:
            raise ValueError(
                "Internal error: Precision and Recall have mismatched shapes: "
                f"{p_tensor.shape} vs {r_tensor.shape}. Please, open an issue "
                "with a reference on this error. Thank you!"
            )
        dict_obj = {}
        for idx, p_label in enumerate(p_tensor):
            dict_obj[_get_label_for_class(idx)] = {
                "precision": p_label.item(),
                "recall": r_tensor[idx].item(),
                "f{0}-score".format(beta): f_tensor[idx].item(),
            }
        dict_obj["macro avg"] = {
            "precision": a_precision.item(),
            "recall": a_recall.item(),
            "f{0}-score".format(beta): a_f.item(),
        }
        return dict_obj if output_dict else json.dumps(dict_obj)

    # helper method to get a label for a given class
    def _get_label_for_class(idx: int) -> str:
        return labels[idx] if labels else str(idx)

    return MetricsLambda(_wrapper, recall, precision, fbeta, averaged_recall, averaged_precision, averaged_fbeta,)
