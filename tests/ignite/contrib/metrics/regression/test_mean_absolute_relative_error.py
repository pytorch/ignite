import torch
from pytest import approx, raises

from ignite.exceptions import NotComputableError
from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError


def test_wrong_input_shapes():
    m = MeanAbsoluteRelativeError()

    with raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4, 1)))

    with raises(ValueError):
        m.update((torch.rand(4, 1),
                  torch.rand(4, 1, 2)))

    with raises(ValueError):
        m.update((torch.rand(4, 1, 2),
                  torch.rand(4,)))

    with raises(ValueError):
        m.update((torch.rand(4,),
                  torch.rand(4, 1, 2)))


def test_mean_absolute_relative_error():
    a = torch.rand(4)
    b = torch.rand(4)
    c = torch.rand(4)
    d = torch.rand(4)
    ground_truth = torch.rand(4)

    m = MeanAbsoluteRelativeError()

    m.update((a, ground_truth))
    abs_error_a = torch.sum(torch.abs(ground_truth - a) / torch.abs(ground_truth))
    num_samples_a = a.size()[0]
    sum_error = abs_error_a
    sum_samples = num_samples_a
    MARE_a = sum_error / sum_samples
    assert m.compute() == approx(MARE_a.item())

    m.update((b, ground_truth))
    abs_error_b = torch.sum(torch.abs(ground_truth - b) / torch.abs(ground_truth))
    num_samples_b = b.size()[0]
    sum_error += abs_error_b
    sum_samples += num_samples_b
    MARE_b = sum_error / sum_samples
    assert m.compute() == approx(MARE_b.item())

    m.update((c, ground_truth))
    abs_error_c = torch.sum(torch.abs(ground_truth - c) / torch.abs(ground_truth))
    num_samples_c = c.size()[0]
    sum_error += abs_error_c
    sum_samples += num_samples_c
    MARE_c = sum_error / sum_samples
    assert m.compute() == approx(MARE_c.item())

    m.update((d, ground_truth))
    abs_error_d = torch.sum(torch.abs(ground_truth - d) / torch.abs(ground_truth))
    num_samples_d = d.size()[0]
    sum_error += abs_error_d
    sum_samples += num_samples_d
    MARE_d = sum_error / sum_samples
    assert m.compute() == approx(MARE_d.item())


def test_zero_div():
    a = torch.tensor([2.0, -1.0, -1.0, 2.0])
    ground_truth = torch.tensor([0.0, 0.5, 0.2, 1.0])

    m = MeanAbsoluteRelativeError()
    with raises(NotComputableError):
        m.update((a, ground_truth))


def test_zero_sample():
    m = MeanAbsoluteRelativeError()
    with raises(NotComputableError):
        m.compute()
