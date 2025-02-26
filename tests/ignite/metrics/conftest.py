import pytest
import torch


@pytest.fixture(params=range(14))
def test_data_binary(request):
    return [
        # Binary accuracy on input of shape (N, 1) or (N, )
        (torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)), 1),
        (torch.randint(0, 2, size=(10, 1)), torch.randint(0, 2, size=(10, 1)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50,)), torch.randint(0, 2, size=(50,)), 16),
        (torch.randint(0, 2, size=(50, 1)), torch.randint(0, 2, size=(50, 1)), 16),
        # Binary accuracy on input of shape (N, L)
        (torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
        (torch.randint(0, 2, size=(10, 1, 5)), torch.randint(0, 2, size=(10, 1, 5)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5)), torch.randint(0, 2, size=(50, 5)), 16),
        (torch.randint(0, 2, size=(50, 1, 5)), torch.randint(0, 2, size=(50, 1, 5)), 16),
        # Binary accuracy on input of shape (N, H, W)
        (torch.randint(0, 2, size=(10, 12, 10)), torch.randint(0, 2, size=(10, 12, 10)), 1),
        (torch.randint(0, 2, size=(10, 1, 12, 10)), torch.randint(0, 2, size=(10, 1, 12, 10)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 12, 10)), torch.randint(0, 2, size=(50, 12, 10)), 16),
        (torch.randint(0, 2, size=(50, 1, 12, 10)), torch.randint(0, 2, size=(50, 1, 12, 10)), 16),
        # Corner case with all zeros predictions
        (torch.zeros(size=(10,), dtype=torch.long), torch.randint(0, 2, size=(10,)), 1),
        (torch.zeros(size=(10, 1), dtype=torch.long), torch.randint(0, 2, size=(10, 1)), 1),
    ][request.param]


@pytest.fixture(params=range(14))
def test_data_multiclass(request):
    return [
        # Multiclass input data of shape (N, ) and (N, C)
        (torch.rand(10, 6), torch.randint(0, 6, size=(10,)), 1),
        (torch.rand(10, 4), torch.randint(0, 4, size=(10,)), 1),
        # updated batches
        (torch.rand(50, 6), torch.randint(0, 6, size=(50,)), 16),
        (torch.rand(50, 4), torch.randint(0, 4, size=(50,)), 16),
        # Multiclass input data of shape (N, L) and (N, C, L)
        (torch.rand(10, 5, 8), torch.randint(0, 5, size=(10, 8)), 1),
        (torch.rand(10, 8, 12), torch.randint(0, 8, size=(10, 12)), 1),
        # updated batches
        (torch.rand(50, 5, 8), torch.randint(0, 5, size=(50, 8)), 16),
        (torch.rand(50, 8, 12), torch.randint(0, 8, size=(50, 12)), 16),
        # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
        (torch.rand(10, 5, 18, 16), torch.randint(0, 5, size=(10, 18, 16)), 1),
        (torch.rand(10, 7, 20, 12), torch.randint(0, 7, size=(10, 20, 12)), 1),
        # updated batches
        (torch.rand(50, 5, 18, 16), torch.randint(0, 5, size=(50, 18, 16)), 16),
        (torch.rand(50, 7, 20, 12), torch.randint(0, 7, size=(50, 20, 12)), 16),
        # Corner case with all zeros predictions
        (torch.zeros(size=(10, 6)), torch.randint(0, 6, size=(10,)), 1),
        (torch.zeros(size=(10, 4)), torch.randint(0, 4, size=(10,)), 1),
    ][request.param]


@pytest.fixture(params=range(14))
def test_data_multilabel(request):
    return [
        # Multilabel input data of shape (N, C)
        (torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
        (torch.randint(0, 2, size=(10, 4)), torch.randint(0, 2, size=(10, 4)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5)), torch.randint(0, 2, size=(50, 5)), 16),
        (torch.randint(0, 2, size=(50, 4)), torch.randint(0, 2, size=(50, 4)), 16),
        # Multilabel input data of shape (N, C, L)
        (torch.randint(0, 2, size=(10, 5, 10)), torch.randint(0, 2, size=(10, 5, 10)), 1),
        (torch.randint(0, 2, size=(10, 4, 10)), torch.randint(0, 2, size=(10, 4, 10)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5, 10)), torch.randint(0, 2, size=(50, 5, 10)), 16),
        (torch.randint(0, 2, size=(50, 4, 10)), torch.randint(0, 2, size=(50, 4, 10)), 16),
        # Multilabel input data of shape (N, C, H, W)
        (torch.randint(0, 2, size=(10, 5, 18, 16)), torch.randint(0, 2, size=(10, 5, 18, 16)), 1),
        (torch.randint(0, 2, size=(10, 4, 20, 23)), torch.randint(0, 2, size=(10, 4, 20, 23)), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5, 18, 16)), torch.randint(0, 2, size=(50, 5, 18, 16)), 16),
        (torch.randint(0, 2, size=(50, 4, 20, 23)), torch.randint(0, 2, size=(50, 4, 20, 23)), 16),
        # Corner case with all zeros predictions
        (torch.zeros(size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
        (torch.zeros(size=(10, 4)), torch.randint(0, 2, size=(10, 4)), 1),
    ][request.param]
