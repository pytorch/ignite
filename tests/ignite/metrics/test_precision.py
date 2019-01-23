import pytest
import warnings

from sklearn.metrics import precision_score
from sklearn.exceptions import UndefinedMetricWarning

from ignite.exceptions import NotComputableError
from ignite.metrics import Precision

import torch

torch.manual_seed(12)


def test_no_update():
    precision = Precision()
    with pytest.raises(NotComputableError):
        precision.compute()


def test_binary_wrong_inputs():
    pr = Precision()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        pr.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.arange(0, 10).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # y_pred values are not thresholded to 0, 1 values
        pr.update((torch.rand(10, 1),
                   torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average):
        pr = Precision(average=average)
        y_pred = torch.randint(0, 2, size=(10, 1))
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, ))
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
        y_pred = torch.round(y_pred)
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100,))
        y = torch.randint(0, 2, size=(100,)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.randint(0, 2, size=(10, 5))
        y = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 5))
        y = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 5))
        y = torch.randint(0, 2, size=(100, 1, 5)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.randint(0, 2, size=(10, 12, 10))
        y = torch.randint(0, 2, size=(10, 12, 10)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 12, 10))
        y = torch.randint(0, 2, size=(10, 1, 12, 10)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        pr = Precision(average=average)
        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 12, 10))
        y = torch.randint(0, 2, size=(100, 1, 12, 10)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == 'binary'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_wrong_inputs():
    pr = Precision()

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).type(torch.LongTensor)))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average):
        pr = Precision(average=average)
        y_pred = torch.rand(20, 6)
        y = torch.randint(0, 6, size=(20,)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(10, 4)
        y = torch.randint(0, 4, size=(10, 1)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # 2-classes
        pr.reset()
        y_pred = torch.rand(10, 2)
        y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 3)
        y = torch.randint(0, 3, size=(100,)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.rand(10, 5, 8)
        y = torch.randint(0, 5, size=(10, 8)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(15, 10, 8)
        y = torch.randint(0, 10, size=(15, 8)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 8, 12)
        y = torch.randint(0, 8, size=(100, 12)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.rand(10, 5, 18, 16)
        y = torch.randint(0, 5, size=(10, 18, 16)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(10, 7, 20, 12)
        y = torch.randint(0, 7, size=(10, 20, 12)).type(torch.LongTensor)
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 8, 12, 14)
        y = torch.randint(0, 8, size=(100, 12, 14)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert pr._type == 'multiclass'
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multilabel_wrong_inputs():
    pr = Precision(average=True, is_multilabel=True)

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible y_pred
        pr.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible y
        pr.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


def test_multilabel_input_NC():

    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(20, 5))
        y = torch.randint(0, 2, size=(20, 5)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 4))
        y = torch.randint(0, 2, size=(10, 4)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 4))
        y = torch.randint(0, 2, size=(100, 4)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multilabel_input_NCL():

    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 10))
        y = torch.randint(0, 2, size=(10, 5, 10)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(15, 4, 10))
        y = torch.randint(0, 2, size=(15, 4, 10)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 4, 12))
        y = torch.randint(0, 2, size=(100, 4, 12)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multilabel_input_NCHW():

    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 18, 16))
        y = torch.randint(0, 2, size=(10, 5, 18, 16)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 5, 12, 14))
        y = torch.randint(0, 2, size=(100, 5, 12, 14)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert pr._type == 'multilabel'
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average='samples') == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_incorrect_type():
    # Tests changing of type during training

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.softmax(torch.rand(4, 4), dim=1)
        y = torch.ones(4).type(torch.LongTensor)
        pr.update((y_pred, y))

        y_pred = torch.randint(0, 2, size=(4,))
        y = torch.ones(4).type(torch.LongTensor)

        with pytest.raises(RuntimeError):
            pr.update((y_pred, y))

    _test(average=True)
    _test(average=False)
