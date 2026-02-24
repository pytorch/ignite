from typing import Callable, Union

import torch

from ignite.metrics.epoch_metric import EpochMetric


def matthews_corrcoef_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import matthews_corrcoef

    if y_preds.ndim == 2 and y_targets.ndim == 2:
        y_preds = torch.argmax(y_preds, dim=1)
        y_targets = torch.argmax(y_targets, dim=1)
    elif y_preds.ndim == 2 and y_targets.ndim == 1:
        y_preds = torch.argmax(y_preds, dim=1)
    elif y_preds.ndim == 1 and y_targets.ndim == 2:
        raise ValueError(
            "Incoherent types between input y_pred and stored predictions: y_pred is 1D while y_target is 2D"
        )

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return matthews_corrcoef(y_true, y_pred)


class MatthewsCorrCoef(EpochMetric):
    """
    Compute the Matthews correlation coefficient (MCC).

    The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications.
    It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.

    This metric is suitable for both binary and multiclass classification.
    In the binary case, it is calculated using the entries of the confusion matrix, whereas for multiclass tasks, it is computed as a generalized correlation coefficient.

    In case of multiclass classification with shape (N, C) for y_pred and (N, C) for y, the predicted class is determined by the argmax of y_pred and y.
    In case of multiclass classification with shape (N, C) for y_pred and (N,) for y, the predicted class is determined by the argmax of y_pred and the true class is determined by the value in y.

        Args:
            output_transform: a callable that is used to transform the
                :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
                form expected by the metric. This can be useful if, for example, you have a multi-output model and
                you want to compute the metric with respect to one of the outputs.
                By default, this metric requires the output as ``(x, y)``.
            device: specifies which device updates are accumulated on. Setting the
                metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
                non-blocking. By default, CPU.
            check_compute_fn: if True, compute_fn is run on the first batch of data to ensure there are no issues.
                If issues exist, user is warned that there might be an issue with the compute_fn. Default, True.
            skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
                true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
                Alternatively, ``output_transform`` can be used to handle this.

            Examples:

            .. include:: defaults.rst
                :start-after: :orphan:


            .. testcode::

                y_pred = torch.tensor([+1, +1, +1, -1])
                y_true = torch.tensor([+1, -1, +1, +1])

                matthews_corrcoef = MatthewsCorrCoef()
                matthews_corrcoef.attach(default_evaluator, 'mcc')
                state = default_evaluator.run([[y_pred, y_true]])
                print(state.metrics['mcc'])

            .. testoutput::

                -0.33...

        .. versionadded:: 0.6.0

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        try:
            from sklearn.metrics import matthews_corrcoef  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This metric module requires scikit-learn to be installed.")

        super().__init__(
            matthews_corrcoef_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
            skip_unrolling=skip_unrolling,
        )
