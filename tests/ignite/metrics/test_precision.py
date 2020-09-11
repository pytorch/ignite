import os
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Precision

torch.manual_seed(12)


def test_no_update():
    precision = Precision()
    with pytest.raises(NotComputableError):
        precision.compute()

    precision = Precision(is_multilabel=True, average=True)
    with pytest.raises(NotComputableError):
        precision.compute()


def test_binary_wrong_inputs():
    pr = Precision()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.arange(0, 10).long()))

    with pytest.raises(ValueError):
        # y_pred values are not thresholded to 0, 1 values
        pr.update((torch.rand(10,), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10, 5, 6)).long(), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average):
        pr = Precision(average=average)
        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
        y_pred = torch.round(y_pred)
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100,))
        y = torch.randint(0, 2, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.randint(0, 2, size=(10, 5))
        y = torch.randint(0, 2, size=(10, 5)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 5))
        y = torch.randint(0, 2, size=(10, 1, 5)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 1, 5))
        y = torch.randint(0, 2, size=(100, 1, 5)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W)

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.randint(0, 2, size=(10, 12, 10))
        y = torch.randint(0, 2, size=(10, 12, 10)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 12, 10))
        y = torch.randint(0, 2, size=(10, 1, 12, 10)).long()
        pr.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 12, 10))
        y = torch.randint(0, 2, size=(100, 12, 10)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_wrong_inputs():
    pr = Precision()

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).long()))

    pr = Precision(average=True)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))

    pr = Precision(average=False)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average):
        pr = Precision(average=average)
        y_pred = torch.rand(20, 6)
        y = torch.randint(0, 6, size=(20,)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(10, 4)
        y = torch.randint(0, 4, size=(10,)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # 2-classes
        pr.reset()
        y_pred = torch.rand(10, 2)
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 3)
        y = torch.randint(0, 3, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
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
        y = torch.randint(0, 5, size=(10, 8)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(15, 10, 8)
        y = torch.randint(0, 10, size=(15, 8)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 8, 12)
        y = torch.randint(0, 8, size=(100, 12)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
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
        y = torch.randint(0, 5, size=(10, 18, 16)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.rand(10, 7, 20, 12)
        y = torch.randint(0, 7, size=(10, 20, 12)).long()
        pr.update((y_pred, y))
        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.rand(100, 8, 12, 14)
        y = torch.randint(0, 8, size=(100, 12, 14)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        num_classes = y_pred.shape[1]
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        assert pr._type == "multiclass"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        sk_average_parameter = "macro" if average else None
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
        pr.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible y_pred
        pr.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible y
        pr.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).long()))
        pr.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).long()))


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


def test_multilabel_input_NC():
    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(20, 5))
        y = torch.randint(0, 2, size=(20, 5)).long()
        pr.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 4))
        y = torch.randint(0, 2, size=(10, 4)).long()
        pr.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 4))
        y = torch.randint(0, 2, size=(100, 4)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    pr1 = Precision(is_multilabel=True, average=True)
    pr2 = Precision(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    pr1.update((y_pred, y))
    pr2.update((y_pred, y))
    assert pr1.compute() == pytest.approx(pr2.compute().mean().item())


def test_multilabel_input_NCL():
    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 10))
        y = torch.randint(0, 2, size=(10, 5, 10)).long()
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(15, 4, 10))
        y = torch.randint(0, 2, size=(15, 4, 10)).long()
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 4, 12))
        y = torch.randint(0, 2, size=(100, 4, 12)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    pr1 = Precision(is_multilabel=True, average=True)
    pr2 = Precision(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    pr1.update((y_pred, y))
    pr2.update((y_pred, y))
    assert pr1.compute() == pytest.approx(pr2.compute().mean().item())


def test_multilabel_input_NCHW():
    def _test(average):
        pr = Precision(average=average, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 5, 18, 16))
        y = torch.randint(0, 2, size=(10, 5, 18, 16)).long()
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr.reset()
        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        pr.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        # Batched Updates
        pr.reset()
        y_pred = torch.randint(0, 2, size=(100, 5, 12, 14))
        y = torch.randint(0, 2, size=(100, 5, 12, 14)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_pred = to_numpy_multilabel(y_pred)
        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    pr1 = Precision(is_multilabel=True, average=True)
    pr2 = Precision(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    pr1.update((y_pred, y))
    pr2.update((y_pred, y))
    assert pr1.compute() == pytest.approx(pr2.compute().mean().item())


def test_incorrect_type():
    # Tests changing of type during training

    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.softmax(torch.rand(4, 4), dim=1)
        y = torch.ones(4).long()
        pr.update((y_pred, y))

        y_pred = torch.randint(0, 2, size=(4,))
        y = torch.ones(4).long()

        with pytest.raises(RuntimeError):
            pr.update((y_pred, y))

    _test(average=True)
    _test(average=False)

    pr1 = Precision(is_multilabel=True, average=True)
    pr2 = Precision(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    pr1.update((y_pred, y))
    pr2.update((y_pred, y))
    assert pr1.compute() == pytest.approx(pr2.compute().mean().item())


def test_incorrect_y_classes():
    def _test(average):
        pr = Precision(average=average)

        y_pred = torch.randint(0, 2, size=(10, 4)).float()
        y = torch.randint(4, 5, size=(10,)).long()

        with pytest.raises(ValueError):
            pr.update((y_pred, y))

    _test(average=True)
    _test(average=False)


def _test_distrib_integration_multiclass(device):
    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(average, n_epochs, metric_device):
        n_iters = 60
        s = 16
        n_classes = 7

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.rand(offset * idist.get_world_size(), n_classes).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset, :],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        pr = Precision(average=average, device=metric_device)
        pr.attach(engine, "pr")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "pr" in engine.state.metrics
        res = engine.state.metrics["pr"]
        if isinstance(res, torch.Tensor):
            assert res.device == metric_device
            res = res.cpu().numpy()

        true_res = precision_score(
            y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy(), average="macro" if average else None
        )

        assert pytest.approx(res) == true_res

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
        for metric_device in metric_devices:
            _test(average=True, n_epochs=1, metric_device=metric_device)
            _test(average=True, n_epochs=2, metric_device=metric_device)
            _test(average=False, n_epochs=1, metric_device=metric_device)
            _test(average=False, n_epochs=2, metric_device=metric_device)


def _test_distrib_integration_multilabel(device):

    from ignite.engine import Engine

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(average, n_epochs, metric_device):
        n_iters = 60
        s = 16
        n_classes = 7

        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset * idist.get_world_size(), n_classes, 6, 8)).to(device)
        y_preds = torch.randint(0, 2, size=(offset * idist.get_world_size(), n_classes, 6, 8)).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset, ...],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset, ...],
            )

        engine = Engine(update)

        pr = Precision(average=average, is_multilabel=True)
        pr.attach(engine, "pr")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "pr" in engine.state.metrics
        res = engine.state.metrics["pr"]
        res2 = pr.compute()
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()
            res2 = res2.cpu().numpy()
            assert (res == res2).all()
        else:
            assert res == res2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            true_res = precision_score(
                to_numpy_multilabel(y_true), to_numpy_multilabel(y_preds), average="samples" if average else None
            )

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
        for metric_device in metric_devices:
            _test(average=True, n_epochs=1, metric_device=metric_device)
            _test(average=True, n_epochs=2, metric_device=metric_device)

    if idist.get_world_size() > 1:
        with pytest.warns(
            RuntimeWarning,
            match="Precision/Recall metrics do not work in distributed setting when "
            "average=False and is_multilabel=True",
        ):
            pr = Precision(average=False, is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(4, 3, 6, 8))
        y = torch.randint(0, 2, size=(4, 3, 6, 8)).long()
        pr.update((y_pred, y))
        pr_compute1 = pr.compute()
        pr_compute2 = pr.compute()
        assert len(pr_compute1) == 4 * 6 * 8
        assert (pr_compute1 == pr_compute2).all()


def _test_distrib_accumulator_device(device):
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average, metric_device):
        pr = Precision(average=average, device=metric_device)
        assert pr._device == metric_device
        # Since the shape of the accumulated amount isn't known before the first update
        # call, the internal variables aren't tensors on the right device yet.

        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))

        assert pr._true_positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._true_positives.device), pr._true_positives.device, type(metric_device), metric_device
        )
        assert pr._positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._positives.device), pr._positives.device, type(metric_device), metric_device
        )

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(True, metric_device=metric_device)
        _test(False, metric_device=metric_device)


def _test_distrib_multilabel_accumulator_device(device):
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average, metric_device):
        pr = Precision(is_multilabel=True, average=average, device=metric_device)

        assert pr._device == metric_device
        assert pr._true_positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._true_positives.device), pr._true_positives.device, type(metric_device), metric_device
        )
        assert pr._positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._positives.device), pr._positives.device, type(metric_device), metric_device
        )

        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        pr.update((y_pred, y))

        assert pr._true_positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._true_positives.device), pr._true_positives.device, type(metric_device), metric_device
        )
        assert pr._positives.device == metric_device, "{}:{} vs {}:{}".format(
            type(pr._positives.device), pr._positives.device, type(metric_device), metric_device
        )

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(True, metric_device=metric_device)
        _test(False, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device("cuda:{}".format(local_rank))
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(local_rank, distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration_multiclass, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_multilabel, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_multilabel, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device("cuda:{}".format(distributed_context_multi_node_nccl["local_rank"]))
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
