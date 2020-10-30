from typing import Callable, Optional, Union

import torch

from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["BalancedAccuracy"]


def BalancedAccuracy(
    precision: Optional[Precision] = None,
    recall: Optional[Recall] = None,
    output_transform: Optional[Callable] = None,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> MetricsLambda:
    """Calculates class balanced accuracy, defined here as follows:
    The minimum between the precision and the recall for each class is computed.
    Those values are then averaged over the total number of classes to get the balanced accuracy.

    Args:
        precision (Precision, optional): precision object metric with `average=False`, used to compute balanced accuracy
        recall (Precision, optional): recall object metric with `average=False`, used to compute balanced accuracy
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device (str or torch.device): specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Returns:
        MetricsLambda, balanced accuracy metric
    """
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

    min_pr = [min(value) for value in zip(precision, recall)]
    balanced_accuracy = sum(min_pr) / len(min_pr)

    return balanced_accuracy
