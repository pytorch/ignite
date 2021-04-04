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
    with pytest.raises(NotComputableError, match=r"Precision must have at least one example before it can be computed"):
        precision.compute()

    precision = Precision(is_multilabel=True, average=True)
    with pytest.raises(NotComputableError, match=r"Precision must have at least one example before it can be computed"):
        precision.compute()


def test_binary_wrong_inputs():
    pr = Precision()

    with pytest.raises(ValueError, match=r"For binary cases, y must be comprised of 0's and 1's"):
        # y has not only 0 or 1 values
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.arange(0, 10).long()))

    with pytest.raises(ValueError, match=r"For binary cases, y_pred must be comprised of 0's and 1's"):
        # y_pred values are not thresholded to 0, 1 values
        pr.update((torch.rand(10,), torch.randint(0, 2, size=(10,)).long(),))

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10, 5, 6)).long(), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))


@pytest.mark.parametrize("average", [False, True])
def test_binary_input_N(average):
    # Binary accuracy on input of shape (N, 1) or (N, )

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100,)), torch.randint(0, 2, size=(100,)), 1),
            (torch.randint(0, 2, size=(100, 1)), torch.randint(0, 2, size=(100, 1)), 1),
            (torch.randint(0, 2, size=(200,)), torch.randint(0, 2, size=(200,)), 1),
            (torch.randint(0, 2, size=(200, 1)), torch.randint(0, 2, size=(200, 1)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100,)), torch.randint(0, 2, size=(100,)), 16),
            (torch.randint(0, 2, size=(100, 1)), torch.randint(0, 2, size=(100, 1)), 16),
            (torch.randint(0, 2, size=(200,)), torch.randint(0, 2, size=(200,)), 16),
            (torch.randint(0, 2, size=(200, 1)), torch.randint(0, 2, size=(200, 1)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y, y_pred, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_binary_input_NL(average):
    # Binary accuracy on input of shape (N, L)

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100, 5)), torch.randint(0, 2, size=(100, 5)), 1),
            (torch.randint(0, 2, size=(100, 1, 5)), torch.randint(0, 2, size=(100, 1, 5)), 1),
            (torch.randint(0, 2, size=(200, 5)), torch.randint(0, 2, size=(200, 5)), 1),
            (torch.randint(0, 2, size=(200, 1, 5)), torch.randint(0, 2, size=(200, 1, 5)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100, 5)), torch.randint(0, 2, size=(100, 5)), 16),
            (torch.randint(0, 2, size=(100, 1, 5)), torch.randint(0, 2, size=(100, 1, 5)), 16),
            (torch.randint(0, 2, size=(200, 5)), torch.randint(0, 2, size=(200, 5)), 16),
            (torch.randint(0, 2, size=(200, 1, 5)), torch.randint(0, 2, size=(200, 1, 5)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y, y_pred, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_binary_input_NHW(average):
    # Binary accuracy on input of shape (N, H, W)

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "binary"
        assert isinstance(pr.compute(), float if average else torch.Tensor)
        pr_compute = pr.compute() if average else pr.compute().numpy()
        assert precision_score(np_y, np_y_pred, average="binary") == pytest.approx(pr_compute)

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100, 12, 10)), torch.randint(0, 2, size=(100, 12, 10)), 1),
            (torch.randint(0, 2, size=(100, 1, 12, 10)), torch.randint(0, 2, size=(100, 1, 12, 10)), 1),
            (torch.randint(0, 2, size=(200, 12, 10)), torch.randint(0, 2, size=(200, 12, 10)), 1),
            (torch.randint(0, 2, size=(200, 1, 12, 10)), torch.randint(0, 2, size=(200, 1, 12, 10)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100, 12, 10)), torch.randint(0, 2, size=(100, 12, 10)), 16),
            (torch.randint(0, 2, size=(100, 1, 12, 10)), torch.randint(0, 2, size=(100, 1, 12, 10)), 16),
            (torch.randint(0, 2, size=(200, 12, 10)), torch.randint(0, 2, size=(200, 12, 10)), 16),
            (torch.randint(0, 2, size=(200, 1, 12, 10)), torch.randint(0, 2, size=(200, 1, 12, 10)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


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


@pytest.mark.parametrize("average", [False, True])
def test_multiclass_input_N(average):
    # Multiclass input data of shape (N, ) and (N, C)

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

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

    def get_test_cases():

        test_cases = [
            (torch.rand(100, 6), torch.randint(0, 6, size=(100,)), 1),
            (torch.rand(100, 4), torch.randint(0, 4, size=(100,)), 1),
            (torch.rand(200, 6), torch.randint(0, 6, size=(200,)), 1),
            (torch.rand(200, 4), torch.randint(0, 4, size=(200,)), 1),
            # updated batches
            (torch.rand(100, 6), torch.randint(0, 6, size=(100,)), 16),
            (torch.rand(100, 4), torch.randint(0, 4, size=(100,)), 16),
            (torch.rand(200, 6), torch.randint(0, 6, size=(200,)), 16),
            (torch.rand(200, 4), torch.randint(0, 4, size=(200,)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_multiclass_input_NL(average):
    # Multiclass input data of shape (N, L) and (N, C, L)

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

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

    def get_test_cases():

        test_cases = [
            (torch.rand(100, 5, 8), torch.randint(0, 5, size=(100, 8)), 1),
            (torch.rand(100, 8, 12), torch.randint(0, 8, size=(100, 12)), 1),
            (torch.rand(200, 5, 8), torch.randint(0, 5, size=(200, 8)), 1),
            (torch.rand(200, 8, 12), torch.randint(0, 8, size=(200, 12)), 1),
            # updated batches
            (torch.rand(100, 5, 8), torch.randint(0, 5, size=(100, 8)), 16),
            (torch.rand(100, 8, 12), torch.randint(0, 8, size=(100, 12)), 16),
            (torch.rand(200, 5, 8), torch.randint(0, 5, size=(200, 8)), 16),
            (torch.rand(200, 8, 12), torch.randint(0, 8, size=(200, 12)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_multiclass_input_NHW(average):
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)

    pr = Precision(average=average)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

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

    def get_test_cases():

        test_cases = [
            (torch.rand(100, 5, 18, 16), torch.randint(0, 5, size=(100, 18, 16)), 1),
            (torch.rand(100, 7, 20, 12), torch.randint(0, 7, size=(100, 20, 12)), 1),
            (torch.rand(200, 5, 18, 16), torch.randint(0, 5, size=(200, 18, 16)), 1),
            (torch.rand(200, 7, 20, 12), torch.randint(0, 7, size=(200, 20, 12)), 1),
            # updated batches
            (torch.rand(100, 5, 18, 16), torch.randint(0, 5, size=(100, 18, 16)), 16),
            (torch.rand(100, 7, 20, 12), torch.randint(0, 7, size=(100, 20, 12)), 16),
            (torch.rand(200, 5, 18, 16), torch.randint(0, 5, size=(200, 18, 16)), 16),
            (torch.rand(200, 7, 20, 12), torch.randint(0, 7, size=(200, 20, 12)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


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


@pytest.mark.parametrize("average", [False, True])
def test_multilabel_input_NC(average):

    pr = Precision(average=average, is_multilabel=True)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y_pred = y_pred.numpy()
        np_y = y.numpy()

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr1 = Precision(is_multilabel=True, average=True)
        pr2 = Precision(is_multilabel=True, average=False)
        pr1.update((y_pred, y))
        pr2.update((y_pred, y))
        assert pr1.compute() == pytest.approx(pr2.compute().mean().item())

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100, 5)), torch.randint(0, 2, size=(100, 5)), 1),
            (torch.randint(0, 2, size=(100, 4)), torch.randint(0, 2, size=(100, 4)), 1),
            (torch.randint(0, 2, size=(200, 5)), torch.randint(0, 2, size=(200, 5)), 1),
            (torch.randint(0, 2, size=(200, 4)), torch.randint(0, 2, size=(200, 4)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100, 5)), torch.randint(0, 2, size=(100, 5)), 16),
            (torch.randint(0, 2, size=(100, 4)), torch.randint(0, 2, size=(100, 4)), 16),
            (torch.randint(0, 2, size=(200, 5)), torch.randint(0, 2, size=(200, 5)), 16),
            (torch.randint(0, 2, size=(200, 4)), torch.randint(0, 2, size=(200, 4)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_multilabel_input_NCL(average):

    pr = Precision(average=average, is_multilabel=True)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr1 = Precision(is_multilabel=True, average=True)
        pr2 = Precision(is_multilabel=True, average=False)
        pr1.update((y_pred, y))
        pr2.update((y_pred, y))
        assert pr1.compute() == pytest.approx(pr2.compute().mean().item())

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100, 5, 10)), torch.randint(0, 2, size=(100, 5, 10)), 1),
            (torch.randint(0, 2, size=(100, 4, 10)), torch.randint(0, 2, size=(100, 4, 10)), 1),
            (torch.randint(0, 2, size=(200, 5, 10)), torch.randint(0, 2, size=(200, 5, 10)), 1),
            (torch.randint(0, 2, size=(200, 4, 10)), torch.randint(0, 2, size=(200, 4, 10)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100, 5, 10)), torch.randint(0, 2, size=(100, 5, 10)), 16),
            (torch.randint(0, 2, size=(100, 4, 10)), torch.randint(0, 2, size=(100, 4, 10)), 16),
            (torch.randint(0, 2, size=(200, 5, 10)), torch.randint(0, 2, size=(200, 5, 10)), 16),
            (torch.randint(0, 2, size=(200, 4, 10)), torch.randint(0, 2, size=(200, 4, 10)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


@pytest.mark.parametrize("average", [False, True])
def test_multilabel_input_NCHW(average):

    pr = Precision(average=average, is_multilabel=True)

    def _test(y_pred, y, batch_size):
        pr.reset()
        pr.update((y_pred, y))

        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert pr._type == "multilabel"
        pr_compute = pr.compute() if average else pr.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average="samples") == pytest.approx(pr_compute)

        pr1 = Precision(is_multilabel=True, average=True)
        pr2 = Precision(is_multilabel=True, average=False)
        pr1.update((y_pred, y))
        pr2.update((y_pred, y))
        assert pr1.compute() == pytest.approx(pr2.compute().mean().item())

    def get_test_cases():

        test_cases = [
            (torch.randint(0, 2, size=(100, 5, 18, 16)), torch.randint(0, 2, size=(100, 5, 18, 16)), 1),
            (torch.randint(0, 2, size=(100, 4, 20, 23)), torch.randint(0, 2, size=(100, 4, 20, 23)), 1),
            (torch.randint(0, 2, size=(200, 5, 18, 16)), torch.randint(0, 2, size=(200, 5, 18, 16)), 1),
            (torch.randint(0, 2, size=(200, 4, 20, 23)), torch.randint(0, 2, size=(200, 4, 20, 23)), 1),
            # updated batches
            (torch.randint(0, 2, size=(100, 5, 18, 16)), torch.randint(0, 2, size=(100, 5, 18, 16)), 16),
            (torch.randint(0, 2, size=(100, 4, 20, 23)), torch.randint(0, 2, size=(100, 4, 20, 23)), 16),
            (torch.randint(0, 2, size=(200, 5, 18, 16)), torch.randint(0, 2, size=(200, 5, 18, 16)), 16),
            (torch.randint(0, 2, size=(200, 4, 20, 23)), torch.randint(0, 2, size=(200, 4, 20, 23)), 16),
        ]

        return test_cases

    for _ in range(10):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


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

        pr = Precision(average=average, is_multilabel=True, device=metric_device)
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

        np_y_preds = to_numpy_multilabel(y_preds)
        np_y_true = to_numpy_multilabel(y_true)
        assert pr._type == "multilabel"
        res = res if average else res.mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y_true, np_y_preds, average="samples") == pytest.approx(res)

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
        for metric_device in metric_devices:
            _test(average=True, n_epochs=1, metric_device=metric_device)
            _test(average=True, n_epochs=2, metric_device=metric_device)
            _test(average=False, n_epochs=1, metric_device=metric_device)
            _test(average=False, n_epochs=2, metric_device=metric_device)

    pr1 = Precision(is_multilabel=True, average=True)
    pr2 = Precision(is_multilabel=True, average=False)
    y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
    pr1.update((y_pred, y))
    pr2.update((y_pred, y))
    assert pr1.compute() == pytest.approx(pr2.compute().mean().item())


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

        assert (
            pr._true_positives.device == metric_device
        ), f"{type(pr._true_positives.device)}:{pr._true_positives.device} vs {type(metric_device)}:{metric_device}"
        assert (
            pr._positives.device == metric_device
        ), f"{type(pr._positives.device)}:{pr._positives.device} vs {type(metric_device)}:{metric_device}"

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
        assert (
            pr._true_positives.device == metric_device
        ), f"{type(pr._true_positives.device)}:{pr._true_positives.device} vs {type(metric_device)}:{metric_device}"
        assert (
            pr._positives.device == metric_device
        ), f"{type(pr._positives.device)}:{pr._positives.device} vs {type(metric_device)}:{metric_device}"

        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        pr.update((y_pred, y))

        assert (
            pr._true_positives.device == metric_device
        ), f"{type(pr._true_positives.device)}:{pr._true_positives.device} vs {type(metric_device)}:{metric_device}"
        assert (
            pr._positives.device == metric_device
        ), f"{type(pr._positives.device)}:{pr._positives.device} vs {type(metric_device)}:{metric_device}"

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
    device = torch.device(f"cuda:{local_rank}")
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
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
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
