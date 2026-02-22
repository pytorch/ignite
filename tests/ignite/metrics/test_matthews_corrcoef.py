from unittest.mock import patch

import pytest
import sklearn
import torch
from sklearn.metrics import matthews_corrcoef

from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import MatthewsCorrCoef

torch.manual_seed(12)


@pytest.fixture()
def mock_no_sklearn():
    with patch.dict("sys.modules", {"sklearn.metrics": None}):
        yield sklearn


def test_no_sklearn(mock_no_sklearn):
    with pytest.raises(ModuleNotFoundError, match=r"This metric module requires scikit-learn to be installed."):
        MatthewsCorrCoef()


def test_no_update():
    mcc = MatthewsCorrCoef()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        mcc.compute()


def test_input_types():
    mcc = MatthewsCorrCoef()
    mcc.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    mcc.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        mcc.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        mcc.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        mcc.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    mcc = MatthewsCorrCoef()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        mcc._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        mcc._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        mcc._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


@pytest.fixture(params=range(4))
def test_data_binary(request):
    return [
        # Binary input data of shape (N,) or (N, 1)
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(2))
def test_binary_input(n_times, test_data_binary, available_device):
    y_pred, y, batch_size = test_data_binary
    mcc = MatthewsCorrCoef(device=available_device)
    assert mcc._device == torch.device(available_device)

    mcc.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            mcc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        mcc.update((y_pred, y))

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    assert isinstance(mcc.compute(), float)
    assert matthews_corrcoef(np_y, np_y_pred) == pytest.approx(mcc.compute())


@pytest.fixture(params=range(2))
def test_data_multiclass(request):
    return [
        # Multiclass input data of shape (N,)
        (torch.randint(0, 5, size=(10,)).long(), torch.randint(0, 5, size=(10,)).long(), 1),
        # updated batches
        (torch.randint(0, 5, size=(50,)).long(), torch.randint(0, 5, size=(50,)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(2))
def test_multiclass_input(n_times, test_data_multiclass, available_device):
    y_pred, y, batch_size = test_data_multiclass
    mcc = MatthewsCorrCoef(device=available_device)
    assert mcc._device == torch.device(available_device)

    mcc.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            mcc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        mcc.update((y_pred, y))

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    assert isinstance(mcc.compute(), float)
    assert matthews_corrcoef(np_y, np_y_pred) == pytest.approx(mcc.compute())


def test_integration(available_device):
    y_pred = torch.tensor([1, 0, 1, 1])
    y_true = torch.tensor([1, 1, 0, 1])

    def update_fn(engine, batch):
        return y_pred, y_true

    evaluator = Engine(update_fn)
    mcc = MatthewsCorrCoef(device=available_device)
    mcc.attach(evaluator, "mcc")

    state = evaluator.run([None], max_epochs=1)

    assert state.metrics["mcc"] == pytest.approx(matthews_corrcoef(y_true.numpy(), y_pred.numpy()))
