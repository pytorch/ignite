import pytest
import warnings

from sklearn.metrics import recall_score
from sklearn.exceptions import UndefinedMetricWarning

from ignite.exceptions import NotComputableError
from ignite.metrics import Recall

import torch

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
                   torch.arange(0, 10).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # y_reed values are not thresholded to 0, 1 values
        re.update((torch.rand(10, 1),
                   torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)),
                   torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10, 5, 6)),
                   torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)),
                   torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average):
        re = Recall(average=average)
        y_reed = torch.randint(0, 2, size=(10, 1))
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
        y_reed = torch.round(y_reed)
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100,))
        y = torch.randint(0, 2, size=(100,)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.randint(0, 2, size=(10, 5))
        y = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(10, 1, 5))
        y = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re = Recall(average=average)
        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100, 5))
        y = torch.randint(0, 2, size=(100, 1, 5)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W)

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.randint(0, 2, size=(10, 12, 10))
        y = torch.randint(0, 2, size=(10, 12, 10)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(10, 1, 12, 10))
        y = torch.randint(0, 2, size=(10, 1, 12, 10)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

        re = Recall(average=average)
        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100, 12, 10))
        y = torch.randint(0, 2, size=(100, 1, 12, 10)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_reed = y_reed.numpy().ravel()
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_reed, average='binary') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_wrong_inputs():
    re = Recall()

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).type(torch.LongTensor)))

    re = Recall(average=True)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))

    re = Recall(average=False)

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).type(torch.LongTensor)))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average):
        re = Recall(average=average)
        y_reed = torch.rand(20, 6)
        y = torch.randint(0, 6, size=(20,)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.rand(10, 4)
        y = torch.randint(0, 4, size=(10, 1)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # 2-classes
        re.reset()
        y_reed = torch.rand(10, 2)
        y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.rand(100, 3)
        y = torch.randint(0, 3, size=(100,)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx: idx + batch_size], y[idx: idx + batch_size]))

        num_classes = y_reed.shape[1]
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.rand(10, 5, 8)
        y = torch.randint(0, 5, size=(10, 8)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average=sk_average_parameter) == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.rand(15, 10, 8)
        y = torch.randint(0, 10, size=(15, 8)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.rand(100, 8, 12)
        y = torch.randint(0, 8, size=(100, 12)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_reed.shape[1]
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.rand(10, 5, 18, 16)
        y = torch.randint(0, 5, size=(10, 18, 16)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.rand(10, 7, 20, 12)
        y = torch.randint(0, 7, size=(10, 20, 12)).type(torch.LongTensor)
        re.update((y_reed, y))
        num_classes = y_reed.shape[1]
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.rand(100, 10, 12, 14)
        y = torch.randint(0, 10, size=(100, 12, 14)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        num_classes = y_reed.shape[1]
        np_y = y.numpy().ravel()
        np_y_reed = y_reed.argmax(dim=1).numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sk_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_reed, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)


def test_multilabel_wrong_inputs():
    re = Recall(average=True, is_multilabel=True)

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible y_reed
        re.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible y
        re.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).type(torch.LongTensor)))
        re.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).type(torch.LongTensor)))


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


def test_multilabel_input_NC():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_reed = torch.randint(0, 2, size=(20, 5))
        y = torch.randint(0, 2, size=(20, 5)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = to_numpy_multilabel(y_reed)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(10, 4))
        y = torch.randint(0, 2, size=(10, 4)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = y_reed.numpy()
        np_y = y.numpy()
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100, 4))
        y = torch.randint(0, 2, size=(100, 4)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy()
        np_y_reed = y_reed.numpy()
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_reed = torch.randint(0, 2, size=(10, 4))
    y = torch.randint(0, 2, size=(10, 4)).type(torch.LongTensor)
    re1.update((y_reed, y))
    re2.update((y_reed, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_multilabel_input_NCL():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_reed = torch.randint(0, 2, size=(10, 5, 10))
        y = torch.randint(0, 2, size=(10, 5, 10)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = to_numpy_multilabel(y_reed)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(15, 4, 10))
        y = torch.randint(0, 2, size=(15, 4, 10)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = to_numpy_multilabel(y_reed)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100, 4, 12))
        y = torch.randint(0, 2, size=(100, 4, 12)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_reed = to_numpy_multilabel(y_reed)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_reed = torch.randint(0, 2, size=(10, 4, 20))
    y = torch.randint(0, 2, size=(10, 4, 20)).type(torch.LongTensor)
    re1.update((y_reed, y))
    re2.update((y_reed, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_multilabel_input_NCHW():

    def _test(average):
        re = Recall(average=average, is_multilabel=True)

        y_reed = torch.randint(0, 2, size=(10, 5, 18, 16))
        y = torch.randint(0, 2, size=(10, 5, 18, 16)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = to_numpy_multilabel(y_reed)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        re.reset()
        y_reed = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
        re.update((y_reed, y))
        np_y_reed = to_numpy_multilabel(y_reed)
        np_y = to_numpy_multilabel(y)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

        # Batched Updates
        re.reset()
        y_reed = torch.randint(0, 2, size=(100, 5, 12, 14))
        y = torch.randint(0, 2, size=(100, 5, 12, 14)).type(torch.LongTensor)

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            re.update((y_reed[idx:idx + batch_size], y[idx:idx + batch_size]))

        np_y = to_numpy_multilabel(y)
        np_y_reed = to_numpy_multilabel(y_reed)
        assert re._type == 'multilabel'
        re_compute = re.compute() if average else re.compute().mean().item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_reed, average='samples') == pytest.approx(re_compute)

    for _ in range(5):
        _test(average=True)
        _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_reed = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
    re1.update((y_reed, y))
    re2.update((y_reed, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_incorrect_type():
    # Tests changing of type during training

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.softmax(torch.rand(4, 4), dim=1)
        y = torch.ones(4).type(torch.LongTensor)
        re.update((y_reed, y))

        y_reed = torch.zeros(4, 1)
        y = torch.ones(4).type(torch.LongTensor)

        with pytest.raises(RuntimeError):
            re.update((y_reed, y))

    _test(average=True)
    _test(average=False)

    re1 = Recall(is_multilabel=True, average=True)
    re2 = Recall(is_multilabel=True, average=False)
    y_reed = torch.randint(0, 2, size=(10, 4, 20, 23))
    y = torch.randint(0, 2, size=(10, 4, 20, 23)).type(torch.LongTensor)
    re1.update((y_reed, y))
    re2.update((y_reed, y))
    assert re1.compute() == pytest.approx(re2.compute().mean().item())


def test_incorrect_y_classes():

    def _test(average):
        re = Recall(average=average)

        y_reed = torch.randint(0, 2, size=(10, 4)).float()
        y = torch.randint(4, 5, size=(10,)).long()

        with pytest.raises(ValueError):
            re.update((y_reed, y))

    _test(average=True)
    _test(average=False)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib(local_rank, distributed_context_single_node):

    def test_distrib_itegration_multiclass():

        import torch.distributed as dist
        from ignite.engine import Engine

        torch.manual_seed(12)
        device = "cuda:{}".format(local_rank)

        def _test(average, n_epochs):
            n_iters = 100
            s = 16
            n_classes = 10

            offset = n_iters * s
            y_true = torch.randint(0, n_classes, size=(offset * dist.get_world_size(), )).to(device)
            y_preds = torch.rand(offset * dist.get_world_size(), n_classes).to(device)

            def update(engine, i):
                return y_preds[i * s + local_rank * offset:(i + 1) * s + local_rank * offset, :], \
                    y_true[i * s + local_rank * offset:(i + 1) * s + local_rank * offset]

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

        for _ in range(5):
            _test(average=True, n_epochs=1)
            _test(average=True, n_epochs=2)
            _test(average=False, n_epochs=1)
            _test(average=False, n_epochs=2)

    test_distrib_itegration_multiclass()

    def test_distrib_itegration_multilabel():

        import torch.distributed as dist
        from ignite.engine import Engine

        torch.manual_seed(12)
        device = "cuda:{}".format(local_rank)

        def _test(average, n_epochs):
            n_iters = 100
            s = 16
            n_classes = 10

            offset = n_iters * s
            y_true = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 10, 12)).to(device)
            y_preds = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 10, 12)).to(device)

            def update(engine, i):
                return y_preds[i * s + local_rank * offset:(i + 1) * s + local_rank * offset, ...], \
                    y_true[i * s + local_rank * offset:(i + 1) * s + local_rank * offset, ...]

            engine = Engine(update)

            re = Recall(average=average, is_multilabel=True, device=device)
            re.attach(engine, "re")

            data = list(range(n_iters))
            engine.run(data=data, max_epochs=n_epochs)

            assert "re" in engine.state.metrics
            res = engine.state.metrics['re']
            if isinstance(res, torch.Tensor):
                res = res.cpu().numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                true_res = recall_score(to_numpy_multilabel(y_true),
                                        to_numpy_multilabel(y_preds),
                                        average='samples' if average else None)

            assert pytest.approx(res) == true_res

        for _ in range(5):
            _test(average=True, n_epochs=1)
            _test(average=True, n_epochs=2)

        with pytest.warns(RuntimeWarning, match="Precision/Recall metrics do not work in distributed setting when "
                                                "average=False and is_multilabel=True"):
            re = Recall(average=False, is_multilabel=True, device=device)

        y_reed = torch.randint(0, 2, size=(10, 5, 18, 16))
        y = torch.randint(0, 2, size=(10, 5, 18, 16)).long()
        re.update((y_reed, y))
        re_compute = re.compute()
        assert len(re_compute) == 10 * 18 * 16

    test_distrib_itegration_multilabel()
