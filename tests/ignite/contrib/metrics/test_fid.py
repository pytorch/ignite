from pathlib import Path

import numpy as np
import PIL
import pytest
import torch
from torchvision.transforms.functional import to_tensor

from ignite.contrib.metrics import FID

try:
    from scipy import linalg
except ImportError:
    raise RuntimeError("This contrib test requires scipy to be installed.")


@pytest.fixture()
def img_filepath():
    fp = Path(__file__).parent / "assets" / "fid_sample.jpg"
    assert fp.exists()
    yield fp.as_posix()


def test_fid():

    size = 10

    pil_img = PIL.Image.open(img_filepath)
    tensor_img = to_tensor(pil_img)

    y = [tensor_img for _ in range(size)]
    y_pred = [(tensor_img + torch.rand_like(tensor_img) * 0.01) for _ in range(size)]

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
