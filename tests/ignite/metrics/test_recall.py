import os

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Recall

import pytest
import warnings

from sklearn.metrics import recall_score
from sklearn.exceptions import UndefinedMetricWarning

torch.manual_seed(12)


def test_no_update():
    recall = Recall()
    with pytest.raises(NotComputableError):
        recall.compute()

    recall = Recall(is_multilabel=True, average=True)
    with pytest.raises(NotComputableError):
        recall.compute()


def test_binary_wrong_inputs():
    re = Recall()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        re.update((torch.randint(0, 2, size=(10,)),
                   torch.arange(0, 10).long()))

    with pytest.raises(ValueError):
        # y_pred values are not thresholded to 0, 1 values
        re.update((torch.rand(10, 1),
                   torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)),
                   torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10, 5, 6)),
                   torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)),
                   torch.randint(0, 2, size=(10, 5, 6)).long()))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average):
        re = Recall(average=average)
        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
        y_pred = torch.round(y_pred)
        y = torch.randint(0, 2, size=(10,)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100,))
        y = torch.randint(0, 2, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.randint(0, 2, size=(10, 5))
        y = torch.randint(0, 2, size=(10, 5)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 5))
        y = torch.randint(0, 2, size=(10, 1, 5)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100, 5))
        y = torch.randint(0, 2, size=(100, 5)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.randint(0, 2, size=(10, 12, 10))
        y = torch.randint(0, 2, size=(10, 12, 10)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 12, 10))
        y = torch.randint(0, 2, size=(10, 1, 12, 10)).long()
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100, 12, 10))
        y = torch.randint(0, 2, size=(100, 12, 10)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_wrong_inputs():
    re = Recall()

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).long()))

    re = Recall(average=True)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))

    re = Recall(average=False)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average):
        re = Recall(average=average)
        y_pred = torch.rand(20, 6)
        y = torch.randint(0, 6, size=(20,)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(10, 4)
        y = torch.randint(0, 4, size=(10,)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # 2-classes
        re.reset()
        y_pred = torch.rand(10, 2)
        y = torch.randint(0, 2, size=(10,)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.rand(100, 3)
        y = torch.randint(0, 3, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.rand(10, 5, 8)
        y = torch.randint(0, 5, size=(10, 8)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sk_average_parameter) == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(15, 10, 8)
        y = torch.randint(0, 10, size=(15, 8)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.rand(100, 8, 12)
        y = torch.randint(0, 8, size=(100, 12)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.rand(10, 5, 18, 16)
        y = torch.randint(0, 5, size=(10, 18, 16)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(10, 7, 20, 12)
        y = torch.randint(0, 7, size=(10, 20, 12)).long()
        re.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.rand(100, 10, 12, 14)
        y = torch.randint(0, 10, size=(100, 12, 14)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multilabel_wrong_inputs():
    re = Recall(average=True, is_multilabel=True)

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible y_pred
        re.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible y
        re.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).long()))
        re.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).long()))


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


def test_multilabel_input_NC():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(20, 5))
        y = torch.randint(0, 2, size=(20, 5)).long()
        re.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(10, 4))
        y = torch.randint(0, 2, size=(10, 4)).long()
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100, 4))
        y = torch.randint(0, 2, size=(100, 4)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4))
    y = torch.randint(0, 2, size=(10, 4)).long()
    re1.update((y_pred, y))
    re2.update((y_pred, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_multilabel_input_NCL():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 10))
        y = torch.randint(0, 2, size=(10, 5, 10)).long()
        re.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(15, 4, 10))
        y = torch.randint(0, 2, size=(15, 4, 10)).long()
        re.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100, 4, 12))
        y = torch.randint(0, 2, size=(100, 4, 12)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20))
    y = torch.randint(0, 2, size=(10, 4, 20)).long()
    re1.update((y_pred, y))
    re2.update((y_pred, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_multilabel_input_NCHW():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 18, 16))
        y = torch.randint(0, 2, size=(10, 5, 18, 16)).long()
        re.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        re.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_pred = torch.randint(0, 2, size=(100, 5, 12, 14))
        y = torch.randint(0, 2, size=(100, 5, 12, 14)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    re1.update((y_pred, y))
    re2.update((y_pred, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_incorrect_type():
    # Tests changing of type during training

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.softmax(torch.rand(4, 4), dim=1)
        y = torch.ones(4).long()
        re.update((y_pred, y))

        y_pred = torch.zeros(4,)
        y = torch.ones(4).long()

        with pytest.raises(RuntimeError):
            re.update((y_pred, y))

    _test(average=True)
    _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    re1.update((y_pred, y))
    re2.update((y_pred, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_incorrect_y_classes():

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.randint(0, 2, size=(10, 4)).float()
        y = torch.randint(4, 5, size=(10,)).long()

        with pytest.raises(ValueError):
            re.update((y_pred, y))

    _test(average=True)
    _test(average=False)


def _test_distrib_itegration_multiclass(device):
    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    torch.manual_seed(12)

    def _test(average, n_epochs):
        n_iters = 60
        s = 16
        n_classes = 7

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * dist.get_world_size(), )).to(device)
        y_preds = torch.rand(offset * dist.get_world_size(), n_classes).to(device)

        def update(engine, i):
            return y_preds[i * s + rank * offset:(i + 1) * s + rank * offset, :], \
                y_true[i * s + rank * offset:(i + 1) * s + rank * offset]

        engine = Engine(update)

        re = Recall(average=average, device=device)
        re.attach(engine, "re")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "re" in engine.state.metrics
        res = engine.state.metrics['re']
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = recall_score(y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy(),
                                average='macro' if average else None)

        assert pytest.approx(res) == true_res

    for _ in range(2):
        _test(average=True, n_epochs=1)
        _test(average=True, n_epochs=2)
        _test(average=False, n_epochs=1)
        _test(average=False, n_epochs=2)


def _test_distrib_itegration_multilabel(device):

    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    torch.manual_seed(12)

    def _test(average, n_epochs):
        n_iters = 60
        s = 16
        n_classes = 7

        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 6, 8)).to(device)
        y_preds = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 6, 8)).to(device)

        def update(engine, i):
            return y_preds[i * s + rank * offset:(i + 1) * s + rank * offset, ...], \
                y_true[i * s + rank * offset:(i + 1) * s + rank * offset, ...]

        engine = Engine(update)

        re = Recall(average=average, is_multilabel=True, device=device)
        re.attach(engine, "re")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "re" in engine.state.metrics
        res = engine.state.metrics['re']
        res2 = re.compute()
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()
            res2 = res2.cpu().numpy()
            assert (res == res2).all()
        else:
            assert res == res2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            true_res = recall_score(to_numpy_multilabel(y_true),
                                    to_numpy_multilabel(y_preds),
                                    average='samples' if average else None)

        assert pytest.approx(res) == true_res

    for _ in range(2):
        _test(average=True, n_epochs=1)
        _test(average=True, n_epochs=2)

    with pytest.warns(RuntimeWarning, match="Precision/Recall metrics do not work in distributed setting when "
                                            "average=False and is_multilabel=True"):
        re = Recall(average=False, is_multilabel=True, device=device)

    y_pred = torch.randint(0, 2, size=(4, 3, 6, 8))
    y = torch.randint(0, 2, size=(4, 3, 6, 8)).long()
    re.update((y_pred, y))
    re_compute1 = re.compute()
    re_compute2 = re.compute()
    assert len(re_compute1) == 4 * 6 * 8
    assert (re_compute1 == re_compute2).all()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)
