from typing import Any, Callable, cast, Tuple, Union

import torch

from ignite import distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetric


def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)


def roc_auc_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> Tuple[Any, Any, Any]:
    from sklearn.metrics import roc_curve

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
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
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

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

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        try:
            from sklearn.metrics import roc_auc_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

        super(ROC_AUC, self).__init__(
            roc_auc_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
            skip_unrolling=skip_unrolling,
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
        device: optional device specification for internal storage.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

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
            Thresholds [inf, 1.0, 0.711, 0.047]

    ..  versionchanged:: 0.4.11
        added `device` argument

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        try:
            from sklearn.metrics import roc_curve  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

        super(RocCurve, self).__init__(
            roc_auc_curve_compute_fn,  # type: ignore[arg-type]
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
            skip_unrolling=skip_unrolling,
        )

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("RocCurve must have at least one example before it can be computed.")

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        ws = idist.get_world_size()
        if ws > 1:
            # All gather across all processes
            _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
            _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))

        if idist.get_rank() == 0:
            # Run compute_fn on zero rank only
            fpr, tpr, thresholds = cast(Tuple, self.compute_fn(_prediction_tensor, _target_tensor))
            fpr = torch.tensor(fpr, dtype=self._double_dtype, device=_prediction_tensor.device)
            tpr = torch.tensor(tpr, dtype=self._double_dtype, device=_prediction_tensor.device)
            thresholds = torch.tensor(thresholds, dtype=self._double_dtype, device=_prediction_tensor.device)
        else:
            fpr, tpr, thresholds = None, None, None

        if ws > 1:
            # broadcast result to all processes
            fpr = idist.broadcast(fpr, src=0, safe_mode=True)
            tpr = idist.broadcast(tpr, src=0, safe_mode=True)
            thresholds = idist.broadcast(thresholds, src=0, safe_mode=True)

        return fpr, tpr, thresholds
