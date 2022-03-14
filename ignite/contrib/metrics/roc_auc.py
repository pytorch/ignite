from typing import Any, Callable, Tuple, Union

import torch

from ignite.metrics import EpochMetric


def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)


def roc_auc_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> Tuple[Any, Any, Any]:
    from sklearn.metrics import roc_curve

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return roc_curve(y_true, y_pred)


class ROC_AUC(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.roc_auc_score <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: Default False. If True, `roc_curve
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#
            sklearn.metrics.roc_auc_score>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
        device: optional device specification for internal storage.

    Note:

        ROC_AUC expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
        values. To apply an activation to y_pred, use output_transform as shown below:

        .. code-block:: python

            def sigmoid_output_transform(output):
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y
            avg_precision = ROC_AUC(sigmoid_output_transform)

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            roc_auc = ROC_AUC()
            #The ``output_transform`` arg of the metric can be used to perform a sigmoid on the ``y_pred``.
            roc_auc.attach(default_evaluator, 'roc_auc')
            y_pred = torch.tensor([[0.0474], [0.5987], [0.7109], [0.9997]])
            y_true = torch.tensor([[0], [0], [1], [0]])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['roc_auc'])

        .. testoutput::

            0.6666...
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        try:
            from sklearn.metrics import roc_auc_score  # noqa: F401
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        super(ROC_AUC, self).__init__(
            roc_auc_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device
        )


class RocCurve(EpochMetric):
    """Compute Receiver operating characteristic (ROC) for binary classification task
    by accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.roc_curve <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: Default False. If True, `sklearn.metrics.roc_curve
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#
            sklearn.metrics.roc_curve>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
    Note:
        RocCurve expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
        values. To apply an activation to y_pred, use output_transform as shown below:

        .. code-block:: python

            def sigmoid_output_transform(output):
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y
            avg_precision = RocCurve(sigmoid_output_transform)

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            roc_auc = RocCurve()
            #The ``output_transform`` arg of the metric can be used to perform a sigmoid on the ``y_pred``.
            roc_auc.attach(default_evaluator, 'roc_auc')
            y_pred = torch.tensor([0.0474, 0.5987, 0.7109, 0.9997])
            y_true = torch.tensor([0, 0, 1, 0])
            state = default_evaluator.run([[y_pred, y_true]])
            print("FPR", [round(i, 3) for i in state.metrics['roc_auc'][0].tolist()])
            print("TPR", [round(i, 3) for i in state.metrics['roc_auc'][1].tolist()])
            print("Thresholds", [round(i, 3) for i in state.metrics['roc_auc'][2].tolist()])

        .. testoutput::

            FPR [0.0, 0.333, 0.333, 1.0]
            TPR [0.0, 0.0, 1.0, 1.0]
            Thresholds [2.0, 1.0, 0.711, 0.047]
    """

    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:

        try:
            from sklearn.metrics import roc_curve  # noqa: F401
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        super(RocCurve, self).__init__(
            roc_auc_curve_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )
