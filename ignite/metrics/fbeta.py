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
    device: Union[str, torch.device] = torch.device("cpu"),
) -> MetricsLambda:
    r"""Calculates F-beta score.

    .. math::
        F_\beta = \left( 1 + \beta^2 \right) * \frac{ \text{precision} * \text{recall} }
        { \left( \beta^2 * \text{precision} \right) + \text{recall} }

    where :math:`\beta` is a positive real factor.

    Args:
        beta (float): weight of precision in harmonic mean
        average (bool, optional): if True, F-beta score is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with F-beta score for each class in multiclass case.
        precision (Precision, optional): precision object metric with `average=False` to compute F-beta score
        recall (Precision, optional): recall object metric with `average=False` to compute F-beta score
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device (str or torch.device): specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Returns:
        MetricsLambda, F-beta metric
    """
    if not (beta > 0):
        raise ValueError(f"Beta should be a positive integer, but given {beta}")

    if precision is not None and output_transform is not None:
        raise ValueError("If precision argument is provided, output_transform should be None")

    if recall is not None and output_transform is not None:
        raise ValueError("If recall argument is provided, output_transform should be None")

    if precision is None:
        precision = Precision(
            output_transform=(lambda x: x) if output_transform is None else output_transform,  # type: ignore[arg-type]
            average=False,
            device=device,
        )
    elif precision._average:
        raise ValueError("Input precision metric should have average=False")

    if recall is None:
        recall = Recall(
            output_transform=(lambda x: x) if output_transform is None else output_transform,  # type: ignore[arg-type]
            average=False,
            device=device,
        )
    elif recall._average:
        raise ValueError("Input recall metric should have average=False")

    fbeta = (1.0 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
