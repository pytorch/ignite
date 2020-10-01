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
        self,
        output_transform: Callable = lambda x: x,
        fid_model: nn.Module = None,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        if fid_model is None:
            try:
                from torchvision import models

                fid_model = models.inception_v3(pretrained=True, transform_input=True)
            except ImportError:
                raise ValueError("Argument fid_model should be set")

        super(FID, self).__init__(output_transform=output_transform, device=device)
        self._fid_model = fid_model.eval().to(self._device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_of_data = 0

        self._features_sum_real = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._features_sum_fake = torch.tensor(0, device=self._device, dtype=torch.float32)

        self._cov_real = None
        self._cov_fake = None

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        with torch.no_grad():
            batch_features_real = self._fid_model(y).sum(axis=0)
            batch_features_fake = self._fid_model(y_pred).sum(axis=0)

        self._num_of_data += 1

        if self._cov_real is None:
            self._cov_real = torch.eye(len(batch_features_real))
        else:
            self._cov_real = self._update_cov(
                batch_features_real, self._features_sum_real, self._num_of_data, self._cov_real
            )

        if self._cov_fake is None:
            self._cov_fake = torch.eye(len(batch_features_fake))
        else:
            self._cov_fake = self._update_cov(
                batch_features_fake, self._features_sum_fake, self._num_of_data, self._cov_fake
            )

        self._features_sum_real = self._features_sum_real + batch_features_real
        self._features_sum_fake = self._features_sum_fake + batch_features_fake

    @sync_all_reduce("_features_sum_real", "_features_sum_fake", "_cov_real", "_cov_fake", "_num_of_data")
    def compute(self) -> Union[torch.Tensor, float]:
        mu_real = self._features_sum_real / self._num_of_data
        sigma_real = self._cov_real

        mu_fake = self._features_sum_fake / self._num_of_data
        sigma_fake = self._cov_fake

        return self._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    def _update_cov(self, batch_features, features_sum, num_of_data, cov_old):
        X = batch_features - (features_sum / num_of_data)

        return torch.ger(X, X) * (num_of_data / (num_of_data + 1) ** 2) + cov_old * num_of_data / (num_of_data + 1)

    def _frechet_distance(self, mu, cov, mu2, cov2):
        cc, _ = linalg.sqrtm(torch.matmul(cov, cov2), disp=False)
        dist = torch.sum((mu - mu2) ** 2) + torch.trace(cov + cov2 - 2 * cc)
        return torch.real(dist)
