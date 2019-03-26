from ignite.exceptions import NotComputableError
from ignite.metrics import RootMeanSquaredError
import pytest
import torch


def test_zero_div():
    rmse = RootMeanSquaredError()
    with pytest.raises(NotComputableError):
        rmse.compute()


def test_compute():
    rmse = RootMeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 2.0

    rmse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    rmse.update((y_pred, y))
    assert isinstance(rmse.compute(), float)
    assert rmse.compute() == 3.0
