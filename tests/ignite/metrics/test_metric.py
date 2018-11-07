from ignite.metrics import Metric
from ignite.engine import State
import torch
from mock import MagicMock
from pytest import approx


def test_no_transform():
    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)

    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            assert output == (y_pred, y)

    metric = DummyMetric()
    state = State(output=(y_pred, y))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_transform():
    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)

    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            assert output == (y_pred, y)

    def transform(output):
        pred_dict, target_dict = output
        return pred_dict['y'], target_dict['y']

    metric = DummyMetric(output_transform=transform)
    state = State(output=({'y': y_pred}, {'y': y}))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


def test_no_grad():
    y_pred = torch.zeros(4, requires_grad=True)
    y = torch.zeros(4, requires_grad=False)

    class DummyMetric(Metric):
        def reset(self):
            pass

        def compute(self):
            pass

        def update(self, output):
            y_pred, y = output
            mse = torch.pow(y_pred - y.view_as(y_pred), 2)
            assert y_pred.requires_grad
            assert not mse.requires_grad

    metric = DummyMetric()
    state = State(output=(y_pred, y))
    engine = MagicMock(state=state)
    metric.iteration_completed(engine)


class ListGatherMetric(Metric):

        def __init__(self, index):
            self.index = index

        def reset(self):
            self.list_ = []

        def update(self, output):
            self.list_ = output

        def compute(self):
            return self.list_[self.index]


def test_arithmetics():
    m0 = ListGatherMetric(0)
    m1 = ListGatherMetric(1)
    m2 = ListGatherMetric(2)

    # __add__
    m0_plus_m1 = m0 + m1
    m0_plus_m1.update([1, 10, 100])
    assert m0_plus_m1.compute() == 11
    m0_plus_m1.update([2, 20, 200])
    assert m0_plus_m1.compute() == 22

    m2_plus_2 = m2 + 2
    m2_plus_2.update([1, 10, 100])
    assert m2_plus_2.compute() == 102

    # __sub__
    m0_minus_m1 = m0 - m1
    m0_minus_m1.update([1, 10, 100])
    assert m0_minus_m1.compute() == -9
    m0_minus_m1.update([2, 20, 200])
    assert m0_minus_m1.compute() == -18

    m2_minus_2 = m2 - 2
    m2_minus_2.update([1, 10, 100])
    assert m2_minus_2.compute() == 98

    # __mul__
    m0_times_m1 = m0 * m1
    m0_times_m1.update([1, 10, 100])
    assert m0_times_m1.compute() == 10
    m0_times_m1.update([2, 20, 200])
    assert m0_times_m1.compute() == 40

    m2_times_2 = m2 * 2
    m2_times_2.update([1, 10, 100])
    assert m2_times_2.compute() == 200

    # __pow__
    m0_pow_m1 = m0 ** m1
    m0_pow_m1.update([1, 10, 100])
    assert m0_pow_m1.compute() == 1
    m0_pow_m1.update([2, 20, 200])
    assert m0_pow_m1.compute() == 2 ** 20

    m2_pow_2 = m2 ** 2
    m2_pow_2.update([1, 10, 100])
    assert m2_pow_2.compute() == 10000

    # __mod__
    m0_mod_m1 = m0 % m1
    m0_mod_m1.update([1, 10, 100])
    assert m0_mod_m1.compute() == 1
    m0_mod_m1.update([2, 20, 200])
    assert m0_mod_m1.compute() == 2

    m2_mod_2 = m2 % 2
    m2_mod_2.update([1, 10, 100])
    assert m2_mod_2.compute() == 0

    # __div__
    m0_div_m1 = m0.__div__(m1)
    m0_div_m1.update([1, 10, 100])
    assert m0_div_m1.compute() == 0
    m0_div_m1.update([2, 20, 200])
    assert m0_div_m1.compute() == 0

    m2_div_2 = m2.__div__(2)
    m2_div_2.update([1, 10, 100])
    assert m2_div_2.compute() == 50

    # __truediv__
    m0_truediv_m1 = m0.__truediv__(m1)
    m0_truediv_m1.update([1, 10, 100])
    assert m0_truediv_m1.compute() == approx(0.1)
    m0_truediv_m1.update([2, 20, 200])
    assert m0_truediv_m1.compute() == approx(0.1)

    m2_truediv_2 = m2.__truediv__(2)
    m2_truediv_2.update([1, 10, 100])
    assert m2_truediv_2.compute() == approx(50.0)

    # __floordiv__
    m0_floordiv_m1 = m0 // m1
    m0_floordiv_m1.update([1, 10, 100])
    assert m0_floordiv_m1.compute() == 0
    m0_floordiv_m1.update([2, 20, 200])
    assert m0_floordiv_m1.compute() == 0

    m2_floordiv_2 = m2 // 2
    m2_floordiv_2.update([1, 10, 100])
    assert m2_floordiv_2.compute() == 50
