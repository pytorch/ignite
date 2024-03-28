from typing import Optional

import numpy as np

import pytest
import torch

import ignite.distributed as idist

from ignite.metrics.regression._base import _BaseRegression, _torch_median


def test_base_regression_shapes():
    class L1(_BaseRegression):
        def reset(self):
            self._sum_of_errors = 0.0

        def _update(self, output):
            y_pred, y = output
            errors = torch.abs(y.view_as(y_pred) - y_pred)
            self._sum_of_errors += torch.sum(errors).item()

        def compute(self):
            return self._sum_of_errors

    m = L1()

    with pytest.raises(ValueError, match=r"Input y_pred should have shape \(N,\) or \(N, 1\)"):
        y = torch.rand([1, 1, 1])
        m.update((y, y))

    with pytest.raises(ValueError, match=r"Input y should have shape \(N,\) or \(N, 1\)"):
        y = torch.rand([1, 1, 1])
        m.update((torch.rand(1, 1), y))

    with pytest.raises(ValueError, match=r"Input data shapes should be the same, but given"):
        m.update((torch.rand(2), torch.rand(2, 1)))

    with pytest.raises(TypeError, match=r"Input y_pred dtype should be float"):
        y = torch.tensor([1, 1])
        m.update((y, y))

    with pytest.raises(TypeError, match=r"Input y dtype should be float"):
        y = torch.tensor([1, 1])
        m.update((y.float(), y))


@pytest.mark.parametrize("size", [100, 101, (30, 3), (31, 3)])
def test_torch_median_numpy(size, device: Optional[str] = None):
    data = torch.rand(size).to(device)
    assert _torch_median(data) == np.median(data.cpu().numpy())


@pytest.mark.tpu
@pytest.mark.parametrize("size", [100, 101, (30, 3), (31, 3)])
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_on_even_size_xla(size):
    device = "xla"
    test_torch_median_numpy(size, device=device)


@pytest.mark.parametrize("size", [100, 101, (30, 3), (31, 3)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_on_even_size_gpu(size):
    test_torch_median_numpy(size, device="cuda")


@pytest.mark.parametrize("size", [100, 101, (30, 3), (31, 3)])
def test_create_even_size_cpu(size):
    test_torch_median_numpy(size, device="cpu")
