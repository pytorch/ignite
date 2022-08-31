import numpy as np

import pytest
import torch

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


def test_torch_median():
    n_iters = 80
    size = 105
    y_true = torch.rand(size=(n_iters * size,))
    y_pred = torch.rand(size=(n_iters * size,))
    e = torch.abs(y_true.view_as(y_pred) - y_pred) / torch.abs(y_true.view_as(y_pred))
    assert _torch_median(e) == np.median(e)

    # Test consistency with torch.quantile and torch.median with odd size
    n_iters = 81
    size = 105
    y_true = torch.rand(size=(n_iters * size,))
    y_pred = torch.rand(size=(n_iters * size,))
    e = torch.abs(y_true.view_as(y_pred) - y_pred) / torch.abs(y_true.view_as(y_pred))
    assert _torch_median(e) == torch.quantile(e, 0.5, interpolation="midpoint")

    n_iters = 81
    size = 105
    y_true = torch.rand(size=(n_iters * size,))
    y_pred = torch.rand(size=(n_iters * size,))
    e = torch.abs(y_true.view_as(y_pred) - y_pred) / torch.abs(y_true.view_as(y_pred))
    assert _torch_median(e) == torch.median(e)
