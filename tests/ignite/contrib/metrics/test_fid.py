from pathlib import Path

import numpy as np
import PIL
import pytest
import torch
from torchvision import models
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


def test_fid(img_filepath):

    size = 10

    pil_img = PIL.Image.open(img_filepath)
    tensor_img = to_tensor(pil_img).view(1, 3, 400, 300)

    y = torch.cat([tensor_img for _ in range(size)])
    y_pred = torch.cat([(tensor_img + torch.rand_like(tensor_img) * 0.01) for _ in range(size)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = models.inception_v3(pretrained=True, transform_input=True).eval().to(devide)

    np_batch_fake = inception_model(y_pred).detach().cpu().numpy()
    np_batch_real = inception_model(y).detach().cpu().numpy()

    mu_fake = np.mean(np_batch_fake, axis=0)
    mu_real = np.mean(np_batch_real, axis=0)

    cov_fake = np.cov(np_batch_fake, rowvar=False)
    cov_real = np.cov(np_batch_real, rowvar=False)

    cc, _ = linalg.sqrtm(np.dot(cov_fake, cov_real), disp=False)
    dist = np.sum((mu_fake - mu_real) ** 2) + np.trace(cov_fake + cov_real - 2 * cc)
    np_fid_value = np.real(dist)

    fid_metric = FID()

    fid_metric.reset()
    fid_metric.update((y_pred, y))
    fid_value = fid_metric.compute()

    assert fid_value == np_fid_value
