from ignite.contrib.metrics import MaximumAbsoluteError
import torch
from pytest import approx


def test_maximum_absolute_error():
    a = torch.tensor([0.0, 0.2, -0.8])
    b = torch.tensor([1.0, -2.0, 3.0])
    c = torch.tensor([0.0, 2.0, 0.0])
    d = torch.tensor([0.0, 0.0, -5.0])
    ground_truth = torch.tensor([0.0, 0.0, 0.0])

    m = MaximumAbsoluteError()
    m.reset()

    m.update((a, ground_truth))
    assert m.compute() == approx(0.8)

    m.update((b, ground_truth))
    assert m.compute() == approx(3.0)

    m.update((c, ground_truth))
    assert m.compute() == approx(3.0)

    m.update((d, ground_truth))
    assert m.compute() == approx(5.0)
