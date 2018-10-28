from ignite.contrib.metrics import MaximumAbsoluteError
import torch


def test_maximum_absolute_error(tol=1e-6):
    a = torch.tensor([0.0, 0.2, -0.8])
    b = torch.tensor([1.0, -2.0, 3.0])
    c = torch.tensor([0.0, 2.0, 0.0])
    d = torch.tensor([0.0, 0.0, -5.0])
    ground_truth = torch.tensor([0.0, 0.0, 0.0])

    m = MaximumAbsoluteError()
    m.reset()

    m.update((a, ground_truth))
    assert abs(m.compute() - 0.8) < tol

    m.update((b, ground_truth))
    assert abs(m.compute() - 3.0) < tol

    m.update((c, ground_truth))
    assert abs(m.compute() - 3.0) < tol

    m.update((d, ground_truth))
    assert abs(m.compute() - 5.0) < tol
