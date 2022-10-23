from typing import Callable, Optional, Union

import torch

from ignite.metrics import EpochMetric


class CohenKappa(EpochMetric):
    """Compute different types of Cohen's Kappa: Non-Wieghted, Linear, Quadratic.
    Accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>`_ .

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        weights: a string is used to define the type of Cohen's Kappa whether Non-Weighted or Linear
            or Quadratic. Default, None.
        check_compute_fn: Default False. If True, `cohen_kappa_score
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>`_
            is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.
        device: optional device specification for internal storage.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = CohenKappa()
            metric.attach(default_evaluator, 'ck')
            y_true = torch.tensor([2, 0, 2, 2, 0, 1])
            y_pred = torch.tensor([0, 0, 2, 2, 0, 2])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['ck'])

        .. testoutput::

            0.4285...

    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        weights: Optional[str] = None,
        check_compute_fn: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):

        try:
            from sklearn.metrics import cohen_kappa_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This contrib module requires scikit-learn to be installed.")
        if weights not in (None, "linear", "quadratic"):
            raise ValueError("Kappa Weighting type must be None or linear or quadratic.")

        # initalize weights
        self.weights = weights

        self.cohen_kappa_compute = self.get_cohen_kappa_fn()

        super(CohenKappa, self).__init__(
            self.cohen_kappa_compute,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
            device=device,
        )

    def get_cohen_kappa_fn(self) -> Callable[[torch.Tensor, torch.Tensor], float]:
        """Return a function computing Cohen Kappa from scikit-learn."""
        from sklearn.metrics import cohen_kappa_score

        def wrapper(y_targets: torch.Tensor, y_preds: torch.Tensor) -> float:
            y_true = y_targets.cpu().numpy()
            y_pred = y_preds.cpu().numpy()
            return cohen_kappa_score(y_true, y_pred, weights=self.weights)

        return wrapper
