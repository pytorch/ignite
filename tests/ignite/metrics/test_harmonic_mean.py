import torch
import pytest
from scipy.stats import hmean
from ignite.metrics import HarmonicMean
from ignite.exceptions import NotComputableError

def test_harmonic_mean_basic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = HarmonicMean(device=device)
    
    data = torch.tensor([1.0, 2.0, 4.0], device=device)
    metric.update(data)
    result = metric.compute()
    
    expected = hmean([1.0, 2.0, 4.0])
    assert result == pytest.approx(expected)

def test_harmonic_mean_multiple_updates():
    metric = HarmonicMean()
    
    metric.update(torch.tensor([1.0, 10.0]))
    
    metric.update(torch.tensor([5.0, 2.0]))
    
    result = metric.compute()
    expected = hmean([1.0, 10.0, 5.0, 2.0])
    assert result == pytest.approx(expected)

def test_harmonic_mean_invalid_input():
    metric = HarmonicMean()
    
    # Test for zero or negative values.
    with pytest.raises(ValueError, match="Harmonic mean is only defined for positive values."):
        metric.update(torch.tensor([1.0, 0.0, -2.0]))

def test_not_computable():
    metric = HarmonicMean()
    with pytest.raises(NotComputableError):
        metric.compute()

def test_reset():
    metric = HarmonicMean()
    metric.update(torch.tensor([1.0, 2.0]))
    metric.reset()
    with pytest.raises(NotComputableError):
        metric.compute()

def test_harmonic_mean_tensor_shape():
    metric = HarmonicMean()

    data = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    metric.update(data)

    result = metric.compute()
    expected = hmean([1.0, 2.0, 4.0, 8.0])
    assert result == pytest.approx(expected)