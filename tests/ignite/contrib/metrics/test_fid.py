import numpy as np
import pytest
import torch
import torchvision
import torchvision.transforms as transforms

from ignite.contrib.metrics import FID

try:
    from scipy import linalg
except ImportError:
    raise RuntimeError("This contrib test requires scipy to be installed.")


def test_fid():

    size = 10

    y = torchvision.datasets.ImageFolder(
        root="assets/fid_sample.jpg", transform=transforms.Compose([transforms.ToTensor()])
    )
    y_pred = torchvision.datasets.ImageFolder(
        root="assets/fid_sample.jpg",
        transform=transforms.compose([transforms.ToTensor(), Lambda(lambda x: x + torch.randn(tensor.size()))]),
    )

    np_y_pred = y_pred.cpu().numpy()
    np_y = y.cpu().numpy()

    mu_fake = np.mean(np_y_pred, axis=0)
    mu_real = np.mean(np_y, axis=0)

    cov_fake = np.cov(np_y_pred, rowvar=False)
    cov_real = np.cov(np_y, rowvar=False)

    cc, _ = linalg.sqrtm(np.dot(cov_fake, cov_real), disp=False)
    dist = np.sum((mu_fake - mu_real) ** 2) + np.trace(cov_fake + cov_real - 2 * cc)
    np_fid_value = np.real(dist)

    fid_metric = FID()

    fid_metric.reset()
    fid_metric.update((y_pred, y))
    fid_value = fid_metric.compute()

    assert fid_value == np_fid_value
