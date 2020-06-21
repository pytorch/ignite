import numpy as np
import pytest
import torch

from ignite.contrib.metrics.regression._base import _BaseRegression, _BaseRegressionEpoch
from ignite.metrics.epoch_metric import EpochMetricWarning


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

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 3), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 7), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 7)))


def test_base_regression_epoch_shapes():
    def compute_fn(y_pred, y):
        return 0.0

    class ZeroEpoch(_BaseRegressionEpoch):
        def __init__(self, output_transform=lambda x: x):
            super(ZeroEpoch, self).__init__(compute_fn, output_transform)

    m = ZeroEpoch()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 3), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 7), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 7)))


def test_base_regression_compute_fn():
    # Wrong compute function
    with pytest.raises(TypeError):
        _BaseRegressionEpoch(12345)


def test_check_compute_fn():
    def compute_fn(y_preds, y_targets):
        raise Exception

    em = _BaseRegressionEpoch(compute_fn, check_compute_fn=True)

    em.reset()
    output1 = (torch.rand(4, 1).float(), torch.randint(0, 2, size=(4, 1), dtype=torch.float32))
    with pytest.warns(EpochMetricWarning, match=r"Probably, there can be a problem with `compute_fn`"):
        em.update(output1)

    em = _BaseRegressionEpoch(compute_fn, check_compute_fn=False)
    em.update(output1)
