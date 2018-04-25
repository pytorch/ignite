from ignite.exceptions import NotComputableError
from ignite.metrics import MeanPairwiseDistance
import pytest
from pytest import approx
import torch


def test_zero_div():
    mpd = MeanPairwiseDistance()
    with pytest.raises(NotComputableError):
        mpd.compute()


def test_compute():
    mpd = MeanPairwiseDistance()

    y_pred = torch.Tensor([[3.0, 4.0], [-3.0, -4.0]])
    y = torch.zeros(2, 2)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(5.0)

    mpd.reset()
    y_pred = torch.Tensor([[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0]])
    y = torch.zeros(2, 4)
    mpd.update((y_pred, y))
    assert isinstance(mpd.compute(), float)
    assert mpd.compute() == approx(8.0)
