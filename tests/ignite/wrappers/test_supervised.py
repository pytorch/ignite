import torch
from torch.nn import Linear
from torch.optim import SGD
from torch.autograd import Variable
from torch.nn.functional import mse_loss

from ignite.wrappers import Supervised


def _build_batch():
    x = torch.FloatTensor([[1.0], [2.0]])
    y = torch.FloatTensor([[3.0], [5.0]])
    return x, y


def _assert_almost_equal(a, b):
    assert round(a - b, 5) == 0


def test_update():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    wrapper = Supervised(model, optimizer, mse_loss)

    _assert_almost_equal(model.weight.data[0, 0], 0.0)
    _assert_almost_equal(model.bias.data[0], 0.0)

    batch = _build_batch()
    loss = wrapper.update(batch)

    _assert_almost_equal(loss, 17.0)
    _assert_almost_equal(model.weight.data[0, 0], 1.3)
    _assert_almost_equal(model.bias.data[0], 0.8)


def test_predict():
    model = Linear(1, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    optimizer = SGD(model.parameters(), 0.1)
    wrapper = Supervised(model, optimizer, mse_loss)

    batch = _build_batch()
    y_pred, y = wrapper.predict(batch)

    _assert_almost_equal(y_pred[0, 0], 0.0)
    _assert_almost_equal(y_pred[1, 0], 0.0)
    _assert_almost_equal(y[0, 0], 3.0)
    _assert_almost_equal(y[1, 0], 5.0)
