import json
from typing import Callable, Collection, Dict, List, Optional, Union

import torch

from ignite.metrics.fbeta import Fbeta
from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["ClassificationReport"]


def ClassificationReport(
    beta: int = 1,
    output_dict: bool = False,
    output_transform: Callable = lambda x: x,
    device: Union[str, torch.device] = torch.device("cpu"),
    is_multilabel: bool = False,
    labels: Optional[List[str]] = None,
) -> MetricsLambda:
    r"""Build a text report showing the main classification metrics. The report resembles in functionality to
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
        is_multilabel: If True, the tensors are assumed to be multilabel.
        device: optional device specification for internal storage.
        labels: Optional list of label indices to include in the report

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Multiclass case

        .. testcode:: 1

            metric = ClassificationReport(output_dict=True)
            metric.attach(default_evaluator, "cr")
            y_true = torch.tensor([2, 0, 2, 1, 0, 1])
            y_pred = torch.tensor([
                [0.0266, 0.1719, 0.3055],
                [0.6886, 0.3978, 0.8176],
                [0.9230, 0.0197, 0.8395],
                [0.1785, 0.2670, 0.6084],
                [0.8448, 0.7177, 0.7288],
                [0.7748, 0.9542, 0.8573],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["cr"].keys())
            print(state.metrics["cr"]["0"])
            print(state.metrics["cr"]["1"])
            print(state.metrics["cr"]["2"])
            print(state.metrics["cr"]["macro avg"])

        .. testoutput:: 1

            dict_keys(['0', '1', '2', 'macro avg'])
            {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.4999...}
            {'precision': 1.0, 'recall': 0.5, 'f1-score': 0.6666...}
            {'precision': 0.3333..., 'recall': 0.5, 'f1-score': 0.3999...}
            {'precision': 0.6111..., 'recall': 0.5, 'f1-score': 0.5222...}

        Multilabel case, the shapes must be (batch_size, num_categories, ...)

        .. testcode:: 2

            metric = ClassificationReport(output_dict=True, is_multilabel=True)
            metric.attach(default_evaluator, "cr")
            y_true = torch.tensor([
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
            ])
            y_pred = torch.tensor([
                [1, 1, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
            ])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["cr"].keys())
            print(state.metrics["cr"]["0"])
            print(state.metrics["cr"]["1"])
            print(state.metrics["cr"]["2"])
            print(state.metrics["cr"]["macro avg"])

        .. testoutput:: 2

            dict_keys(['0', '1', '2', 'macro avg'])
            {'precision': 0.2, 'recall': 1.0, 'f1-score': 0.3333...}
            {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.6666...}
            {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
            {'precision': 0.2333..., 'recall': 0.6666..., 'f1-score': 0.3333...}

    """

    # setup all the underlying metrics
    precision = Precision(average=False, is_multilabel=is_multilabel, output_transform=output_transform, device=device)
    recall = Recall(average=False, is_multilabel=is_multilabel, output_transform=output_transform, device=device)
    fbeta = Fbeta(beta, average=False, precision=precision, recall=recall)
    averaged_precision = precision.mean()
    averaged_recall = recall.mean()
    averaged_fbeta = fbeta.mean()

    def _wrapper(
        re: torch.Tensor, pr: torch.Tensor, f: torch.Tensor, a_re: torch.Tensor, a_pr: torch.Tensor, a_f: torch.Tensor
    ) -> Union[Collection[str], Dict]:
        if pr.shape != re.shape:
            raise ValueError(
                "Internal error: Precision and Recall have mismatched shapes: "
                f"{pr.shape} vs {re.shape}. Please, open an issue "
                "with a reference on this error. Thank you!"
            )
        dict_obj = {}
        for idx, p_label in enumerate(pr):
            dict_obj[_get_label_for_class(idx)] = {
                "precision": p_label.item(),
                "recall": re[idx].item(),
                f"f{beta}-score": f[idx].item(),
            }
        dict_obj["macro avg"] = {
            "precision": a_pr.item(),
            "recall": a_re.item(),
            f"f{beta}-score": a_f.item(),
        }
        return dict_obj if output_dict else json.dumps(dict_obj)

    # helper method to get a label for a given class
    def _get_label_for_class(idx: int) -> str:
        return labels[idx] if labels else str(idx)

    return MetricsLambda(_wrapper, recall, precision, fbeta, averaged_recall, averaged_precision, averaged_fbeta)
