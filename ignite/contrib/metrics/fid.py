import os
from typing import Callable, Sequence, Union

import torch
import torch.nn as nn
from torchvision import transforms

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

try:
    from scipy import linalg
except ImportError:
    raise RuntimeError("This contrib module requires scipy to be installed.")


__all__ = ["FID"]


class FID(Metric):
    """Calculates FID metric

    """

    def __init__(
        self, output_transform: Callable = lambda x: x, fid_model: nn.Module = None,
    ):
        if fid_model is None:
            try:
                from torchvision import models

                fid_model = models.inception_v3(pretrained=True)
            except ImportError:
                raise ValueError("Argument fid_model should be set")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fid_model = fid_model.eval().to(self._devide)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_of_data = 0

        self._features_sum_real = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._features_sum_fake = torch.tensor(0, device=self._device, dtype=torch.float32)

        self._cov_real = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._cov_fake = torch.tensor(0, device=self._device, dtype=torch.float32)

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        with torch.no_grad():
            batch_features_real = self._fid_model(y)
            batch_features_fake = self._fid_model(y_pred)

        batch_features_real = batch_features_real.view(*batch_features_real.size()[:2], -1).sum(-1)
        batch_features_fake = batch_features_fake.view(*batch_features_fake.size()[:2], -1).sum(-1)

        self._num_of_data += 1

        self._features_sum_real += batch_features_real
        self._features_sum_fake += batch_features_fake

        if self._num_of_data > 1:
            X_real = y - (self._features_sum_real / self._num_of_data)
            X_fake = y_pred - (self._features_sum_fake / self._num_of_data)

            self._cov_real = torch.ger(X_real, X_real) * (
                self._num_of_data / (self._num_of_data + 1) ** 2
            ) + self._cov_real * self._num_of_data / (self._num_of_data + 1)
            self._cov_fake = torch.ger(X_fake, X_fake) * (
                self._num_of_data / (self._num_of_data + 1) ** 2
            ) + self._cov_fake * self._num_of_data / (self._num_of_data + 1)
        else:
            self._cov_real = torch.eye(len(y))
            self._cov_fake = torch.eye(len(y_pred))

    @sync_all_reduce
    def compute(self) -> Union[torch.Tensor, float]:
        mu_real = self._features_sum_real / self._num_of_data
        sigma_real = self._cov_real

        mu_fake = self._features_sum_real / self._num_of_data
        sigma_fake = self._cov_fake

        return self._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    def _frechet_distance(mu, cov, mu2, cov2):
        cc, _ = linalg.sqrtm(torch.dot(cov, cov2), disp=False)
        dist = torch.sum((mu - mu2) ** 2) + torch.trace(cov + cov2 - 2 * cc)
        return torch.real(dist)
