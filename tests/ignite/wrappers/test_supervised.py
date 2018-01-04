import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.autograd import Variable
from torch.nn.functional import mse_loss
from pytest import approx

from ignite.wrappers import Supervised


def _build_batch():
    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    return x, y


def test_update():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    wrapper = Supervised(model, optimizer, mse_loss)

    assert model.weight.data[0, 0] == approx(0.0)
    assert model.bias.data[0] == approx(0.0)

    batch = _build_batch()
    loss = wrapper.update(batch)

    assert loss == approx(17.0)
    assert model.weight.data[0, 0] == approx(1.3)
    assert model.bias.data[0] == approx(0.8)


def test_predict():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    wrapper = Supervised(model, optimizer, mse_loss)

    batch = _build_batch()
    y_pred, y = wrapper.predict(batch)

    assert y_pred[0, 0] == approx(0.0)
    assert y_pred[1, 0] == approx(0.0)
    assert y[0, 0] == approx(3.0)
    assert y[1, 0] == approx(5.0)
