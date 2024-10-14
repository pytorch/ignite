from typing import Tuple

from torch import Tensor

from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetric


class _ClusteringMetricBase(EpochMetric):
    required_output_keys = ("features", "labels")

    def _check_shape(self, output: Tuple[Tensor, Tensor]) -> None:
        features, labels = output
        if features.ndimension() != 2:
            raise ValueError("Features should be of shape (batch_size, n_targets).")

        if labels.ndimension() != 1:
            raise ValueError("Labels should be of shape (batch_size, ).")

    def _check_type(self, output: Tuple[Tensor, Tensor]) -> None:
        features, labels = output
        if len(self._predictions) < 1:
            return
        dtype_preds = self._predictions[-1].dtype
        if dtype_preds != features.dtype:
            raise ValueError(
                f"Incoherent types between input features and stored features: {dtype_preds} vs {features.dtype}"
            )

        dtype_targets = self._targets[-1].dtype
        if dtype_targets != labels.dtype:
            raise ValueError(
                f"Incoherent types between input labels and stored labels: {dtype_targets} vs {labels.dtype}"
            )

    def compute(self) -> float:
        if len(self._predictions) < 1 or len(self._targets) < 1:
            raise NotComputableError(
                f"{self.__class__.__name__} must have at least one example before it can be computed."
            )

        return super().compute()
