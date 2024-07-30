from typing import Callable, Union

import torch

from ignite.metrics.epoch_metric import EpochMetric


def average_precision_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    from sklearn.metrics import average_precision_score

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return average_precision_score(y_true, y_pred)


class AveragePrecision(EpochMetric):
    """Computes Average Precision accumulating predictions and the ground-truth during an epoch
    and applying `sklearn.metrics.average_precision_score <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn: Default False. If True, `average_precision_score
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            #sklearn.metrics.average_precision_score>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
        device: optional device specification for internal storage.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Note:
        AveragePrecision expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or
        confidence values. To apply an activation to y_pred, use output_transform as shown below:

        .. code-block:: python

            def activated_output_transform(output):
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred, y
            avg_precision = AveragePrecision(activated_output_transform)

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            y_pred = torch.tensor([[0.79, 0.21], [0.30, 0.70], [0.46, 0.54], [0.16, 0.84]])
            y_true = torch.tensor([[1, 1], [1, 1], [0, 1], [0, 1]])

            avg_precision = AveragePrecision()
            avg_precision.attach(default_evaluator, 'average_precision')
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['average_precision'])

        .. testoutput::

            0.9166...

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
            from sklearn.metrics import average_precision_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")

        super(AveragePrecision, self).__init__(
            average_precision_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
            skip_unrolling=skip_unrolling,
        )
