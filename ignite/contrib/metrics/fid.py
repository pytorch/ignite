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

        self._features_real = []
        self._features_fake = []

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        for (img_fake, img_real) in zip(y_pred, y):
            with torch.no_grad():
                batch_features_real = self._fid_model(img_real.unsqueeze(0))[0]
                batch_features_fake = self._fid_model(img_fake.unsqueeze(0))[0]

            self._features_sum_real.append(batch_features_real)
            self._features_sum_fake.append(batch_features_fake)

            self._num_of_data += 1

            self._features_sum_real = self._features_sum_real + batch_features_real
            self._features_sum_fake = self._features_sum_fake + batch_features_fake

    @sync_all_reduce("_features_sum_real", "_features_sum_fake", "_features_real", "_features_fake", "_num_of_data")
    def compute(self) -> Union[torch.Tensor, float]:
        feature_dim = len(self._features_real[0])

        mu_real = self._features_sum_real / self._num_of_data
        sigma_real = self._cov(torch.cat(self._features_real).view(self._num_of_data, feature_dim))

        mu_fake = self._features_sum_fake / self._num_of_data
        sigma_fake = self._cov(torch.cat(self._features_fake).view(self._num_of_data, feature_dim))

        return self._frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    def _cov(x, rowvar=False):
        # PyTorch implementation of numpy.cov from https://github.com/pytorch/pytorch/issues/19037
        if x.dim() == 1:
            x = x.view(-1, 1)

        avg = torch.mean(x, 0)
        fact = self._num_of_data - 1
        xm = x.sub(avg.expand_as(x))
        X_T = xm.t()
        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()

    def _frechet_distance(self, mu, cov, mu2, cov2):
        cc, _ = linalg.sqrtm(torch.matmul(cov, cov2), disp=False)
        dist = torch.sum((mu - mu2) ** 2) + torch.trace(cov + cov2 - 2 * cc)
        if dist.dtype == torch.cfloat or dist.dtype == torch.cdouble:
            return torch.real(dist)
        else:
            return dist
