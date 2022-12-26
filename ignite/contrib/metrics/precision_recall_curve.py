from typing import Any, Callable, cast, Tuple, Union

import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import EpochMetric


def precision_recall_curve_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> Tuple[Any, Any, Any]:
    try:
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return precision_recall_curve(y_true, y_pred)


class PrecisionRecallCurve(EpochMetric):
    """Compute precision-recall pairs for different probability thresholds for binary classification task
    by accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.precision_recall_curve <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: Default False. If True, `precision_recall_curve
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
            #sklearn.metrics.precision_recall_curve>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    Note:
        PrecisionRecallCurve expects y to be comprised of 0's and 1's. y_pred must either be probability estimates
        or confidence values. To apply an activation to y_pred, use output_transform as shown below:

        .. code-block:: python

            def sigmoid_output_transform(output):
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y
            avg_precision = PrecisionRecallCurve(sigmoid_output_transform)

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            y_pred = torch.tensor([0.0474, 0.5987, 0.7109, 0.9997])
            y_true = torch.tensor([0, 0, 1, 1])
            prec_recall_curve = PrecisionRecallCurve()
            prec_recall_curve.attach(default_evaluator, 'prec_recall_curve')
            state = default_evaluator.run([[y_pred, y_true]])

            print("Precision", [round(i, 4) for i in state.metrics['prec_recall_curve'][0].tolist()])
            print("Recall", [round(i, 4) for i in state.metrics['prec_recall_curve'][1].tolist()])
            print("Thresholds", [round(i, 4) for i in state.metrics['prec_recall_curve'][2].tolist()])

        .. testoutput::

            Precision [1.0, 1.0, 1.0]
            Recall [1.0, 0.5, 0.0]
            Thresholds [0.7109, 0.9997]

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(PrecisionRecallCurve, self).__init__(
            precision_recall_curve_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError("PrecisionRecallCurve must have at least one example before it can be computed.")

        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        ws = idist.get_world_size()
        if ws > 1 and not self._is_reduced:
            # All gather across all processes
            _prediction_tensor = cast(torch.Tensor, idist.all_gather(_prediction_tensor))
            _target_tensor = cast(torch.Tensor, idist.all_gather(_target_tensor))
        self._is_reduced = True

        if idist.get_rank() == 0:
            # Run compute_fn on zero rank only
            precision, recall, thresholds = self.compute_fn(_prediction_tensor, _target_tensor)
            precision = torch.tensor(precision)
            recall = torch.tensor(recall)
            # thresholds can have negative strides, not compatible with torch tensors
            # https://discuss.pytorch.org/t/negative-strides-in-tensor-error/134287/2
            thresholds = torch.tensor(thresholds.copy())
        else:
            precision, recall, thresholds = None, None, None

        if ws > 1:
            # broadcast result to all processes
            precision = idist.broadcast(precision, src=0, safe_mode=True)
            recall = idist.broadcast(recall, src=0, safe_mode=True)
            thresholds = idist.broadcast(thresholds, src=0, safe_mode=True)

        return precision, recall, thresholds
