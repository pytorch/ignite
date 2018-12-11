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


def test_binary_wrong_inputs():
    re = Recall()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        re.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.arange(0, 10).type(torch.LongTensor)))

    # TODO: Uncomment the following after 0.1.2 release
    # with pytest.raises(ValueError):
    #     # y_pred values are not thresholded to 0, 1 values
    #     pr.update((torch.rand(10, 1),
    #                torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                   torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average):
        re = Recall(average=average)
        y_pred = torch.rand(10, 1)
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        # TODO: y_pred should be binary after 0.1.2 release
        # y_pred = torch.randint(0, 2, size=(10, )).type(torch.LongTensor)
        y_pred = torch.rand(10)
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        # TODO: y_pred should be binary after 0.1.2 release
        y_pred = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.51])
        y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

    _test(average=True)
    _test(average=False)


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)

    def _test(average):
        re = Recall(average=average)

        # TODO: y_pred should be binary after 0.1.2 release
        # y_pred = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
        y_pred = torch.rand(10, 5)
        y = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        pr_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

        re.reset()
        # TODO: y_pred should be binary after 0.1.2 release
        # y_pred = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
        y_pred = torch.rand(10, 1, 5)
        y = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        pr_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(pr_compute)

    _test(average=True)
    _test(average=False)


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W)

    def _test(average):
        re = Recall(average=average)

        # TODO: y_pred should be binary after 0.1.2 release
        # y_pred = torch.randint(0, 2, size=(10, 12, 10)).type(torch.LongTensor)
        y_pred = torch.rand(10, 12, 10)
        y = torch.randint(0, 2, size=(10, 12, 10)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

        re.reset()
        # TODO: y_pred should be binary after 0.1.2 release
        # y_pred = torch.randint(0, 2, size=(10, 1, 12, 10)).type(torch.LongTensor)
        y_pred = torch.rand(10, 1, 12, 10)
        y = torch.randint(0, 2, size=(10, 1, 12, 10)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y = y.numpy().ravel()
        # np_y_pred = y_pred.numpy().ravel()
        np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
        assert re._type == 'binary'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        assert recall_score(np_y, np_y_pred, average='binary') == pytest.approx(re_compute)

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


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average):
        re = Recall(average=average)
        y_pred = torch.rand(20, 6)
        y = torch.randint(0, 5, size=(20,)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(10, 4)
        y = torch.randint(0, 3, size=(10, 1)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

        # 2-classes
        re.reset()
        y_pred = torch.rand(10, 2)
        y = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

    _test(average=True)
    _test(average=False)


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.rand(10, 5, 8)
        y = torch.randint(0, 4, size=(10, 8)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(15, 10, 8)
        y = torch.randint(0, 9, size=(15, 8)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

    _test(average=True)
    _test(average=False)


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.rand(10, 5, 18, 16)
        y = torch.randint(0, 4, size=(10, 18, 16)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

        re.reset()
        y_pred = torch.rand(10, 7, 20, 12)
        y = torch.randint(0, 6, size=(10, 20, 12)).type(torch.LongTensor)
        re.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert re._type == 'multiclass'
        assert isinstance(re.compute(), float if average else torch.Tensor)
        re_compute = re.compute() if average else re.compute().numpy()
        sklearn_average_parameter = 'macro' if average else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sklearn_average_parameter) == pytest.approx(re_compute)

    _test(average=True)
    _test(average=False)


def test_incorrect_type():
    # Tests changing of type during training

    def _test(average):
        re = Recall(average=average)

        y_pred = torch.softmax(torch.rand(4, 4), dim=1)
        y = torch.ones(4).type(torch.LongTensor)
        re.update((y_pred, y))

        y_pred = torch.rand(4, 1)
        y = torch.ones(4).type(torch.LongTensor)

        with pytest.raises(RuntimeError):
            re.update((y_pred, y))

    _test(average=True)
    _test(average=False)
