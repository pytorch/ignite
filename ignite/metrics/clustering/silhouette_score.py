from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor

from ignite.metrics.clustering._base import _ClusteringMetricBase

__all__ = ["SilhouetteScore"]


class SilhouetteScore(_ClusteringMetricBase):
    r"""Calculates the
    `silhouette score <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_.

    The silhouette score evaluates the quality of clustering results.

    .. math::
        s = \frac{b-a}{\max(a,b)}

    where:

    - :math:`a` is the mean distance between a sample and all other points in the same cluster.
    - :math:`b` is the mean distance between a sample and all other points in the next nearest cluster.

    More details can be found
    `here <https://scikit-learn.org/1.5/modules/clustering.html#silhouette-coefficient>`_.

    The silhouette score ranges from -1 to +1,
    where the score becomes close to +1 when the clustering result is good (i.e., clusters are well-separated).

    The computation of this metric is implemented with
    `sklearn.metrics.silhouette_score
    <https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.silhouette_score.html>`_.

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
        silhouette_kwargs: additional arguments passed to ``sklearn.metrics.silhouette_score``.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in format of
        ``(features, labels)`` or ``{'features': features, 'labels': labels, ...}``.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = SilhouetteScore()
            metric.attach(default_evaluator, "silhouette_score")
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
            print(state.metrics["silhouette_score"])

        .. testoutput::

            0.12607366

    .. versionadded:: 0.5.2
    """

    def __init__(
        self,
        output_transform: Callable[..., Any] = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
        silhouette_kwargs: Optional[dict] = None,
    ) -> None:
        try:
            from sklearn.metrics import silhouette_score  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scikit-learn to be installed.")

        self._silhouette_kwargs = {} if silhouette_kwargs is None else silhouette_kwargs

        super().__init__(self._silhouette_score, output_transform, check_compute_fn, device, skip_unrolling)

    def _silhouette_score(self, features: Tensor, labels: Tensor) -> float:
        from sklearn.metrics import silhouette_score

        np_features = features.numpy()
        np_labels = labels.numpy()
        score = silhouette_score(np_features, np_labels, **self._silhouette_kwargs)
        return score
