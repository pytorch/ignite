from typing import Any, Callable, Union

import torch
from torch import Tensor

from ignite.metrics.clustering._base import _ClusteringMetricBase

__all__ = ["CalinskiHarabaszScore"]


def _calinski_harabasz_score(features: Tensor, labels: Tensor) -> float:
    from sklearn.metrics import calinski_harabasz_score

    np_features = features.cpu().numpy()
    np_labels = labels.cpu().numpy()
    score = calinski_harabasz_score(np_features, np_labels)
    return score


class CalinskiHarabaszScore(_ClusteringMetricBase):
    r"""Calculates the
    `Calinski-Harabasz score <https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index>`_.

    The Calinski-Harabasz score evaluates the quality of clustering results.

    More details can be found
    `here <https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index>`_.

    A higher Calinski-Harabasz score indicates that
    the clustering result is good (i.e., clusters are well-separated).

    The computation of this metric is implemented with
    `sklearn.metrics.calinski_harabasz_score
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html>`_.

    - ``update`` must receive output of the form ``(features, labels)``
      or ``{'features': features, 'labels': labels}``.
    - `features` and `labels` must be of same shape `(B, D)` and `(B,)`.

    Parameters are inherited from ``EpochMetric.__init__``.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(features, labels)``
            or ``{'features': features, 'labels': labels}``.
        check_compute_fn: if True, ``compute_fn`` is run on the first batch of data to ensure there are no
            issues. If issues exist, user is warned that there might be an issue with the ``compute_fn``.
            Default, True.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(features, labels)`` or ``{'features': features, 'labels': labels, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = CalinskiHarabaszScore()
            metric.attach(default_evaluator, "calinski_harabasz_score")
            X = torch.tensor([
                    [-1.04, -0.71, -1.42, -0.28, -0.43],
                    [0.47, 0.96, -0.43, 1.57, -2.24],
                    [-0.62, -0.29, 0.10, -0.72, -1.69],
                    [0.96, -0.77, 0.60, -0.89, 0.49],
                    [-1.33, -1.53, 0.25, -1.60, -2.0],
                    [-0.63, -0.55, -1.03, -0.89, -0.77],
                    [-0.26, -1.67, -0.24, -1.33, -0.40],
                    [-0.20, -1.34, -0.52, -1.55, -1.50],
                    [2.68, 1.13, 2.51, 0.80, 0.92],
                    [0.33, 2.88, 1.35, -0.56, 1.71]
            ])
            Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
            state = default_evaluator.run([{"features": X, "labels": Y}])
            print(state.metrics["calinski_harabasz_score"])

        .. testoutput::

            5.733936

    .. versionadded:: 0.5.2
    """

    def __init__(
        self,
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        try:
            from sklearn.metrics import calinski_harabasz_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scikit-learn to be installed.")

        super().__init__(_calinski_harabasz_score, output_transform, check_compute_fn, device, skip_unrolling)
