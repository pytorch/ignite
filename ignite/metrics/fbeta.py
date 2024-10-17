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

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...) or (batch_size, ...).
    - `y` must be in the following shape (batch_size, ...).

    Args:
        beta: weight of precision in harmonic mean
        average: if True, F-beta score is computed as the unweighted average (across all classes
            in multiclass case), otherwise, returns a tensor with F-beta score for each class in multiclass case.
        precision: precision object metric with `average=False` to compute F-beta score
        recall: recall object metric with `average=False` to compute F-beta score
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. It is used only if precision or recall are not provided.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Returns:
        MetricsLambda, F-beta metric

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        Binary case

        .. testcode:: 1

            P = Precision(average=False)
            R = Recall(average=False)
            metric = Fbeta(beta=1.0, precision=P, recall=R)
            metric.attach(default_evaluator, "f-beta")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["f-beta"])

        .. testoutput:: 1

            0.7499...

        Multiclass case

        .. testcode:: 2

            P = Precision(average=False)
            R = Recall(average=False)
            metric = Fbeta(beta=1.0, precision=P, recall=R)
            metric.attach(default_evaluator, "f-beta")
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
            print(state.metrics["f-beta"])

        .. testoutput:: 2

            0.5222...

        F-beta can be computed for each class as done below:

        .. testcode:: 3

            P = Precision(average=False)
            R = Recall(average=False)
            metric = Fbeta(beta=1.0, average=False, precision=P, recall=R)
            metric.attach(default_evaluator, "f-beta")
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
            print(state.metrics["f-beta"])

        .. testoutput:: 3

            tensor([0.5000, 0.6667, 0.4000], dtype=torch.float64)

        The elements of `y` and `y_pred` should have 0 or 1 values. Thresholding of predictions can
        be done as below:

        .. testcode:: 4

            def thresholded_output_transform(output):
                y_pred, y = output
                y_pred = torch.round(y_pred)
                return y_pred, y

            P = Precision(average=False, output_transform=thresholded_output_transform)
            R = Recall(average=False, output_transform=thresholded_output_transform)
            metric = Fbeta(beta=1.0, precision=P, recall=R)
            metric.attach(default_evaluator, "f-beta")
            y_true = torch.tensor([1, 0, 1, 1, 0, 1])
            y_pred = torch.tensor([0.6, 0.2, 0.9, 0.4, 0.7, 0.65])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["f-beta"])

        .. testoutput:: 4

            0.7499...
    """
    if not (beta > 0):
        raise ValueError(f"Beta should be a positive integer, but given {beta}")

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

    fbeta = (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
