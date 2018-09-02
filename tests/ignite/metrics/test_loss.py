from ignite.exceptions import NotComputableError
from ignite.metrics import Loss
import pytest
import torch
from torch import nn
from torch.nn.functional import nll_loss
from numpy.testing import assert_almost_equal


def test_zero_div():
    loss = Loss(nll_loss)
    with pytest.raises(NotComputableError):
        loss.compute()


def test_compute():
    loss = Loss(nll_loss)

    y_pred = torch.Tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.LongTensor([2, 2])
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1512925625)

    y_pred = torch.Tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log()
    y = torch.LongTensor([2, 0, 2])
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1253643036)  # average


def test_compute_on_criterion():
    loss = Loss(nn.NLLLoss())

    y_pred = torch.Tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.LongTensor([2, 2])
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1512925625)

    y_pred = torch.Tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2], [0.2, 0.7, 0.1]]).log()
    y = torch.LongTensor([2, 0, 2])
    loss.update((y_pred, y))
    assert_almost_equal(loss.compute(), 1.1253643036)  # average


def test_non_averaging_loss():
    loss = Loss(nn.NLLLoss(reduction='none'))

    y_pred = torch.Tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]]).log()
    y = torch.LongTensor([2, 2])
    with pytest.raises(AssertionError):
        loss.update((y_pred, y))


def test_reset():
    loss = Loss(nll_loss)

    y_pred = torch.Tensor([[0.1, 0.3, 0.6], [0.6, 0.2, 0.2]]).log()
    y = torch.LongTensor([2, 0])
    loss.update((y_pred, y))
    loss.compute()
    loss.reset()
    with pytest.raises(NotComputableError):
        loss.compute()
