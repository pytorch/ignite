from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy
import pytest
import torch
from sklearn.metrics import accuracy_score


torch.manual_seed(12)


def test_no_update():
    acc = Accuracy()
    with pytest.raises(NotComputableError):
        acc.compute()


def test__check_shape():
    acc = Accuracy()

    # Check squeezed dimensions
    y_pred, y = acc._check_shape((torch.randint(0, 2, size=(10, 1, 5, 6)).type(torch.LongTensor),
                                  torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)))
    assert y_pred.shape == (10, 5, 6)
    assert y.shape == (10, 5, 6)

    y_pred, y = acc._check_shape((torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                                  torch.randint(0, 2, size=(10, 1, 5, 6)).type(torch.LongTensor)))
    assert y_pred.shape == (10, 5, 6)
    assert y.shape == (10, 5, 6)


def test_binary_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        acc.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                    torch.arange(0, 10).type(torch.LongTensor)))

    # TODO: Uncomment the following after 0.1.2 release
    # with pytest.raises(ValueError):
    #     # y_pred values are not thresholded to 0, 1 values
    #     acc.update((torch.rand(10, 1),
    #                 torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                    torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor),
                    torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).type(torch.LongTensor),
                    torch.randint(0, 2, size=(10, 5, 6)).type(torch.LongTensor)))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )
    acc = Accuracy()

    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(10, 1)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(10, )).type(torch.LongTensor)
    y_pred = torch.rand(10)
    y = torch.randint(0, 2, size=(10,)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)
    acc = Accuracy()

    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
    y_pred = torch.rand(10, 5)
    y = torch.randint(0, 2, size=(10, 5)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
    y_pred = torch.rand(10, 1, 5)
    y = torch.randint(0, 2, size=(10, 1, 5)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W, ...)
    acc = Accuracy()

    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(4, 12, 10)).type(torch.LongTensor)
    y_pred = torch.rand(4, 12, 10)
    y = torch.randint(0, 2, size=(4, 12, 10)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(4, 1, 12, 10)).type(torch.LongTensor)
    y_pred = torch.rand(4, 1, 12, 10)
    y = torch.randint(0, 2, size=(4, 1, 12, 10)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    # TODO: y_pred should be binary after 0.1.2 release
    # y_pred = torch.randint(0, 2, size=(4, 12, 10, 8)).type(torch.LongTensor)
    y_pred = torch.rand(4, 12, 10, 8)
    y = torch.randint(0, 2, size=(4, 12, 10, 8)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y = y.numpy().ravel()
    # np_y_pred = y_pred.numpy().ravel()
    np_y_pred = (y_pred.numpy().ravel() > 0.5).astype('int')
    assert acc._type == 'binary'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_multiclass_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 4),
                    torch.randint(0, 2, size=(10,)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 6),
                    torch.randint(0, 5, size=(10, 5)).type(torch.LongTensor)))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10),
                    torch.randint(0, 5, size=(10, 5, 6)).type(torch.LongTensor)))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)

    acc = Accuracy()

    y_pred = torch.rand(10, 4)
    y = torch.randint(0, 4, size=(10,)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(4, 10)
    y = torch.randint(0, 10, size=(4, 1)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # 2-classes
    acc.reset()
    y_pred = torch.rand(4, 2)
    y = torch.randint(0, 2, size=(4, 1)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)
    acc = Accuracy()

    y_pred = torch.rand(10, 4, 5)
    y = torch.randint(0, 4, size=(10, 5)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(4, 10, 5)
    y = torch.randint(0, 10, size=(4, 5)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
    acc = Accuracy()

    y_pred = torch.rand(4, 5, 12, 10)
    y = torch.randint(0, 5, size=(4, 12, 10)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    acc.reset()
    y_pred = torch.rand(4, 5, 10, 12, 8)
    y = torch.randint(0, 5, size=(4, 10, 12, 8)).type(torch.LongTensor)
    acc.update((y_pred, y))
    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()
    assert acc._type == 'multiclass'
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_incorrect_type():
    acc = Accuracy()

    # Start as binary data
    y_pred = torch.rand(4,)
    y = torch.ones(4).type(torch.LongTensor)
    acc.update((y_pred, y))

    # And add a multiclass data
    y_pred = torch.rand(4, 4)
    y = torch.ones(4).type(torch.LongTensor)

    with pytest.raises(RuntimeError):
        acc.update((y_pred, y))
