from typing import Callable

import torch

from ignite.metrics import EpochMetric


def non_weighted_cohen_kappa_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return cohen_kappa_score(y_true, y_pred, weights=None)


class NonWeightedCohenKappa(EpochMetric):
    """Computes Cohen Kappa with no weights. Applying `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>` .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn (bool): Default False. If True, `NonWeightedCohenKappa
            `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
            generated/sklearn.metrics.cohen_kappa_score.html>` is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred, y

        non_weighted_cohen_kappa = NonWeightedCohenKappa(activated_output_transform)

    """
    
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(NonWeightedCohenKappa, self).__init__(
            non_weighted_cohen_kappa_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )


def linear_cohen_kappa_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return cohen_kappa_score(y_true, y_pred, weights='linear')


class LinearCohenKappa(EpochMetric):
    """Computes Cohen Kappa with linear weights. Applying `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>` .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn (bool): Default False. If True, `LinearCohenKappa
            `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
            generated/sklearn.metrics.cohen_kappa_score.html>` is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred, y

        lienar_cohen_kappa = LinearCohenKappa(activated_output_transform)

    """
    
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(LinearCohenKappa, self).__init__(
            linear_cohen_kappa_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )


def quadratic_cohen_kappa_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


class QuadraticCohenKappa(EpochMetric):
    """Computes Cohen Kappa with quadratic weights. Applying `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
    generated/sklearn.metrics.cohen_kappa_score.html>` .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn (bool): Default False. If True, `QuadraticCohenKappa
            `sklearn.metrics.cohen_kappa_score <https://scikit-learn.org/stable/modules/
            generated/sklearn.metrics.cohen_kappa_score.html>` is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.softmax(y_pred, dim=1)
            return y_pred, y

        quadratic_cohen_kappa = QuadraticCohenKappa(activated_output_transform)

    """
    
    def __init__(self, output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(QuadraticCohenKappa, self).__init__(
            quadratic_cohen_kappa_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn
        )
