from ignite.exceptions import NotComputableError
from ignite.metrics import MeanAbsoluteError
import pytest
import torch


def test_zero_div():
    mae = MeanAbsoluteError()
    with pytest.raises(NotComputableError):
        mae.compute()


def test_compute():
    mae = MeanAbsoluteError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    mae.update((y_pred, y))
    assert isinstance(mae.compute(), float)
    assert mae.compute() == 2.0

    mae.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    mae.update((y_pred, y))
    assert isinstance(mae.compute(), float)
    assert mae.compute() == 3.0
