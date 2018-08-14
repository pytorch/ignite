from ignite.metrics import Metric
from ignite.engine import State
import torch
from mock import MagicMock


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
