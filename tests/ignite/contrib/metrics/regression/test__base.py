from typing import Optional
from unittest import mock

import numpy as np

import pytest
import torch
from packaging.version import Version

import ignite.distributed as idist

from ignite.contrib.metrics.regression._base import _BaseRegression, _torch_median


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


@pytest.mark.parametrize("size", [101, (31, 3)])
def test_torch_median_numpy(size, device: Optional[str] = None):
    data = torch.rand(size).to(device)
    assert _torch_median(data) == torch.quantile(data)

    size = 101
    test_tensor = torch.rand(size=(size,))
    assert _torch_median(data) == torch.median(data)


@pytest.mark.tpu
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_on_xla():
    device = "xla"
    test_torch_median_numpy(device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_on_gpu():
    test_torch_median_numpy(device="cuda")


def test_create_supervised_evaluator_on_cpu():
    test_torch_median_numpy(device="cpu")
