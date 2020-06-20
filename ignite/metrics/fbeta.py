from typing import Callable, Optional, Union

import torch

from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["Fbeta"]


def Fbeta(
    beta: float,
    average: bool = True,
    precision: Optional[Precision] = None,
    recall: Optional[Recall] = None,
    output_transform: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> MetricsLambda:
    """Calculates F-beta score

    Args:
        beta (float): weight of precision in harmonic mean
        average (bool, optional): if True, F-beta score is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with F-beta score for each class in multiclass case.
        precision (Precision, optional): precision object metric with `average=False` to compute F-beta score
        recall (Precision, optional): recall object metric with `average=False` to compute F-beta score
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device (str of torch.device, optional): optional device specification for internal storage.

    Returns:
        MetricsLambda, F-beta metric
    """
    if not (beta > 0):
        raise ValueError("Beta should be a positive integer, but given {}".format(beta))

    if precision is not None and output_transform is not None:
        raise ValueError("If precision argument is provided, output_transform should be None")

    if recall is not None and output_transform is not None:
        raise ValueError("If recall argument is provided, output_transform should be None")

    if precision is None:
        precision = Precision(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=device,
        )
    elif precision._average:
        raise ValueError("Input precision metric should have average=False")

    if recall is None:
        recall = Recall(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=device,
        )
    elif recall._average:
        raise ValueError("Input recall metric should have average=False")

    fbeta = (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
