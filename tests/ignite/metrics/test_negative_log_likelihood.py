from ignite.exceptions import NotComputableError
from ignite.metrics import NegativeLogLikelihood
import pytest
import torch
from torch import nn
from numpy.testing import assert_almost_equal


def test_zero_div():
    nll = NegativeLogLikelihood()
    with pytest.raises(NotComputableError):
        nll.compute()


def test_compute():
    nll = NegativeLogLikelihood()

    y_pred = torch.Tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.LongTensor([2, 2])
    nll.update((y_pred, y))
    assert_almost_equal(nll.compute(), 1.1512925625)

    y_pred = torch.Tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log()
    y = torch.LongTensor([2, 0, 2])
    nll.update((y_pred, y))
    assert_almost_equal(nll.compute(), 1.1253643036)  # average


def test_reset():
    nll = NegativeLogLikelihood()

    y_pred = torch.Tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2]]).log()
    y = torch.LongTensor([2, 0])
    nll.update((y_pred, y))
    nll.compute()
    nll.reset()
    with pytest.raises(NotComputableError):
        nll.compute()
