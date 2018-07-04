from ignite.metrics import Metric, EpochMetric
from ignite.engine import State
import torch
from mock import MagicMock
import pytest


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


def test_epoch_metric():

    em = EpochMetric()

    # Wrong input dims
    with pytest.raises(AssertionError):
        output = (torch.tensor(0), torch.tensor(0))
        em.update(output)

    # Wrong input dims
    with pytest.raises(AssertionError):
        output = (torch.rand(4, 3, 1), torch.rand(4, 3))
        em.update(output)

    # Target is not binary
    with pytest.raises(AssertionError):
        output = (torch.rand(4, 3), torch.randint(0, 5, size=(4, 3)))
        em.update(output)

    torch.manual_seed(12)
    em.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)))
    em.update(output1)
    output2 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)))
    em.update(output2)

    assert em._predictions.device.type == 'cpu' and em._targets.device.type == 'cpu'
    assert em._predictions[:4, :] == output1[0]
    assert em._predictions[4:, :] == output2[0]
    assert em._targets[:4, :] == output1[1]
    assert em._targets[4:, :] == output2[1]
