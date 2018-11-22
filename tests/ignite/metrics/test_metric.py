from ignite.metrics import Metric
from ignite.engine import Engine, State
import torch
from mock import MagicMock
from pytest import raises


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


def test_attach():
    class CountMetric(Metric):
        def __init__(self, value):
            self.reset_count = 0
            super(CountMetric, self).__init__()
            self.reset_count = 0
            self.compute_count = 0
            self.update_count = 0
            self.value = value

        def reset(self):
            self.reset_count += 1

        def compute(self):
            self.compute_count += 1
            return self.value

        def update(self, output):
            self.update_count += 1

    def process_function(*args, **kwargs):
        return 1

    engine = Engine(process_function)
    m1 = CountMetric(123)
    m2 = CountMetric(456)
    m1.attach(engine, "m1")
    m2.attach(engine, "m2_1")
    m2.attach(engine, "m2_2")
    engine.run(range(10), 5)

    assert engine.state.metrics["m1"] == 123
    assert engine.state.metrics["m2_1"] == 456
    assert engine.state.metrics["m2_2"] == 456

    assert m1.reset_count == 5
    assert m1.compute_count == 5
    assert m1.update_count == 50

    assert m2.reset_count == 5
    assert m2.compute_count == 10
    assert m2.update_count == 50


def test_abstract_class():
    with raises(TypeError):
        Metric()
