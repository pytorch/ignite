from collections.abc import Callable
from typing import cast

import torch

from ignite.metrics.metrics_lambda import MetricsLambda
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

__all__ = ["Fbeta"]


def Fbeta(
    beta: float,
    average: bool = True,
    precision: Precision | None = None,
    recall: Recall | None = None,
    output_transform: Callable | None = None,
    device: str | torch.device | None = None,
    class_names: list[str] | None = None,
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
        class_names: list of class name strings used to label per-class output. Default: ``None``.

            .. versionadded:: 0.6.0

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

    if precision is not None:
        if output_transform is not None:
            raise ValueError("If precision argument is provided, output_transform should be None")
        if device is not None:
            raise ValueError("If precision argument is provided, device should be None")

    if recall is not None:
        if output_transform is not None:
            raise ValueError("If recall argument is provided, output_transform should be None")
        if device is not None:
            raise ValueError("If recall argument is provided, device should be None")

    if precision is None and recall is None and device is None:
        device = torch.device("cpu")

    if class_names is not None:
        if not isinstance(class_names, (list, tuple)) or not all(isinstance(n, str) for n in class_names):
            raise ValueError("class_names must be a list of strings")
        if average is not False and average is not None:
            raise ValueError(
                f"class_names is only applicable when average=False or average=None, got average={average!r}."
            )

    active_metrics = [m for m in (precision, recall) if m is not None]

    if any(m._average for m in active_metrics):
        raise ValueError("Input precision and recall metrics should have average=False")

    if class_names is not None and any(m._class_names != class_names for m in active_metrics):
        raise ValueError("precision and recall metric class_names must match Fbeta class_names")

    if len(active_metrics) == 2 and active_metrics[0]._class_names != active_metrics[1]._class_names:
        raise ValueError("precision and recall class_names must match")

    target_class_names = class_names
    if target_class_names is None and precision is not None:
        target_class_names = precision._class_names
    if target_class_names is None and recall is not None:
        target_class_names = recall._class_names

    if precision is None:
        precision = Precision(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=cast(str | torch.device, recall._device if recall else device),
            class_names=target_class_names,
        )

    if recall is None:
        recall = Recall(
            output_transform=(lambda x: x) if output_transform is None else output_transform,
            average=False,
            device=cast(str | torch.device, precision._device if precision else device),
            class_names=target_class_names,
        )

    if target_class_names is not None:

        def _fbeta_with_class_names(p: dict, r: dict) -> dict:
            p_vals = torch.tensor(list(p.values()))
            r_vals = torch.tensor(list(r.values()))
            scores = (1.0 + beta**2) * p_vals * r_vals / (beta**2 * p_vals + r_vals + 1e-15)
            if scores.ndim == 0:
                return {target_class_names[0]: scores.item()}
            return dict(zip(target_class_names, scores.tolist()))

        return MetricsLambda(_fbeta_with_class_names, precision, recall)

    fbeta = (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-15)

    if average:
        fbeta = fbeta.mean().item()

    return fbeta
