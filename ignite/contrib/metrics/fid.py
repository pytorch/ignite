import os
from typing import Callable, Sequence, Union

import torch
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
        self, output_transform: Callable = lambda x: x, fid_model=None,
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
        self._value = None
        self._mu_real, self._sigma_real = None, None
        self._mu_fake, self._sigma_fake = None, None

        self._features_real = []
        self._features_fake = []

    @reinit__is_reduced
    def update(self, output) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        data_real, data_fake = y, y_pred
        if isinstance(y_pred, (tuple, list)):
            data_real = y[0]
            data_fake = y_pred[0]
        data_real = self._scale_for_fid(data_real).to(self._device)
        data_fake = self._scale_for_fid(data_fake).to(self._device)
        with torch.no_grad():
            batch_features_real = self._fid_model(data_real)
            batch_features_fake = self._fid_model(data_fake)
        batch_features_real = batch_features_real.view(*batch_features_real.size()[:2], -1).mean(-1)
        batch_features_fake = batch_features_fake.view(*batch_features_fake.size()[:2], -1).mean(-1)

        self._features_real.append(batch_features_real.cpu())
        self._features_fake.append(batch_features_fake.cpu())

    @sync_all_reduce
    def compute(self) -> Union(torch.Tensor, float):
        self._features_real = torch.cat(self._features_real, dim=0)
        self._features_fake = torch.cat(self._features_fake, dim=0)

        self._mu_real = torch.mean(self._features_real, axis=0)
        self._sigma_real = self._cov(self._features_real, rowvar=False)

        self._mu_fake = torch.mean(self._features_fake, axis=0)
        self._sigma_fake = self._cov(self._features_fake, rowvar=False)

        self._value = self._frechet_distance([self._mu_real, self._sigma_real, self._mu_fake, self._sigma_fake])
        return self._value

    def _frechet_distance(mu, cov, mu2, cov2):
        cc, _ = linalg.sqrtm(torch.dot(cov, cov2), disp=False)
        dist = torch.sum((mu - mu2) ** 2) + torch.trace(cov + cov2 - 2 * cc)
        return torch.real(dist)

    def _cov(x, rowvar=False):
        # PyTorch implementation of numpy.cov from https://github.com/pytorch/pytorch/issues/19037
        if x.dim() == 1:
            x = x.view(-1, 1)

        avg = torch.mean(x, 0)
        fact = x.shape[0] - 1
        xm = x.sub(avg.expand_as(x))
        X_T = xm.t()
        c = torch.mm(X_T, xm)
        c = c / fact

        return c.squeeze()
