import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity as ski_ssim

from ignite.exceptions import NotComputableError
from ignite.metrics import SSIM


def test_zero_div():
    ssim = SSIM()
    with pytest.raises(NotComputableError):
        ssim.compute()


def test_ssim():
    ssim = SSIM()
    y_pred = torch.rand(16, 3, 32, 32)
    y = y_pred + 0.125
    ssim.update((y_pred, y))

    np_pred = np.random.rand(16, 32, 32, 3)
    np_y = np.add(np_pred, 0.125)
    np_ssim = ski_ssim(np_pred, np_y, win_size=11, multichannel=True, gaussian_weights=True)

    assert isinstance(ssim.compute(), torch.Tensor)
    assert torch.allclose(ssim.compute(), torch.tensor(np_ssim, dtype=torch.float32), atol=1e-3, rtol=1e-3)
