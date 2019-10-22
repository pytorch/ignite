import os
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

    with pytest.raises(ValueError):
        acc._check_shape((torch.randint(0, 2, size=(10, 1, 5, 12)).long(),
                          torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError):
        acc._check_shape((torch.randint(0, 2, size=(10, 1, 6)).long(),
                          torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError):
        acc._check_shape((torch.randint(0, 2, size=(10, 1)).long(),
                          torch.randint(0, 2, size=(10, 5)).long()))


def test_binary_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError):
        # y has not only 0 or 1 values
        acc.update((torch.randint(0, 2, size=(10,)).long(),
                    torch.arange(0, 10).long()))

    with pytest.raises(ValueError):
        # y_pred values are not thresholded to 0, 1 values
        acc.update((torch.rand(10,),
                    torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).long(),
                    torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10, 5, 6)).long(),
                    torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).long(),
                    torch.randint(0, 2, size=(10, 5, 6)).long()))


def test_binary_input_N():
    # Binary accuracy on input of shape (N, 1) or (N, )
    def _test():
        acc = Accuracy()

        y_pred = torch.randint(0, 2, size=(10,)).long()
        y = torch.randint(0, 2, size=(10,)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100,)).long()
        y = torch.randint(0, 2, size=(100,)).long()

        n_iters = 16
        batch_size = y.shape[0] // n_iters + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_binary_input_NL():
    # Binary accuracy on input of shape (N, L)
    def _test():
        acc = Accuracy()

        y_pred = torch.randint(0, 2, size=(10, 5)).long()
        y = torch.randint(0, 2, size=(10, 5)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(10, 1, 5)).long()
        y = torch.randint(0, 2, size=(10, 1, 5)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 8)).long()
        y = torch.randint(0, 2, size=(100, 8)).long()

        n_iters = 16
        batch_size = y.shape[0] // n_iters + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_binary_input_NHW():
    # Binary accuracy on input of shape (N, H, W, ...)
    def _test():
        acc = Accuracy()

        y_pred = torch.randint(0, 2, size=(4, 12, 10)).long()
        y = torch.randint(0, 2, size=(4, 12, 10)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(4, 1, 12, 10)).long()
        y = torch.randint(0, 2, size=(4, 1, 12, 10)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 8, 8)).long()
        y = torch.randint(0, 2, size=(100, 8, 8)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_binary_as_multiclass_input():
    # Binary accuracy on input of shape (N, 1, ...)
    def _test():
        acc = Accuracy()

        y_pred = torch.randint(0, 2, size=(4, 1)).long()
        y = torch.randint(0, 2, size=(4,)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(4, 1, 12)).long()
        y = torch.randint(0, 2, size=(4, 12)).long()
        acc.update((y_pred, y))
        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 1, 8, 8)).long()
        y = torch.randint(0, 2, size=(100, 8, 8)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()
        assert acc._type == 'binary'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_multiclass_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 4),
                    torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 6),
                    torch.randint(0, 5, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10),
                    torch.randint(0, 5, size=(10, 5, 6)).long()))


def test_multiclass_input_N():
    # Multiclass input data of shape (N, ) and (N, C)
    def _test():
        acc = Accuracy()

        y_pred = torch.rand(10, 4)
        y = torch.randint(0, 4, size=(10,)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.rand(10, 10, 1)
        y = torch.randint(0, 18, size=(10, 1)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.rand(10, 18)
        y = torch.randint(0, 18, size=(10,)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.rand(4, 10)
        y = torch.randint(0, 10, size=(4,)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # 2-classes
        acc.reset()
        y_pred = torch.rand(4, 2)
        y = torch.randint(0, 2, size=(4,)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.rand(100, 5)
        y = torch.randint(0, 5, size=(100,)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_multiclass_input_NL():
    # Multiclass input data of shape (N, L) and (N, C, L)
    def _test():
        acc = Accuracy()

        y_pred = torch.rand(10, 4, 5)
        y = torch.randint(0, 4, size=(10, 5)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.rand(4, 10, 5)
        y = torch.randint(0, 10, size=(4, 5)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.rand(100, 9, 7)
        y = torch.randint(0, 9, size=(100, 7)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_multiclass_input_NHW():
    # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
    def _test():
        acc = Accuracy()

        y_pred = torch.rand(4, 5, 12, 10)
        y = torch.randint(0, 5, size=(4, 12, 10)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.rand(4, 5, 10, 12, 8)
        y = torch.randint(0, 5, size=(4, 10, 12, 8)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        np_y = y.numpy().ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.rand(100, 3, 8, 8)
        y = torch.randint(0, 3, size=(100, 8, 8)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
        assert acc._type == 'multiclass'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


def test_multilabel_wrong_inputs():
    acc = Accuracy(is_multilabel=True)

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible y_pred
        acc.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible y
        acc.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))


def test_multilabel_input_N():
    # Multilabel input data of shape (N, C, ...) and (N, C, ...)

    def _test():
        acc = Accuracy(is_multilabel=True)
        y_pred = torch.randint(0, 2, size=(10, 4))
        y = torch.randint(0, 2, size=(10, 4)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(50, 7)).long()
        y = torch.randint(0, 2, size=(50, 7)).long()
        acc.update((y_pred, y))
        np_y_pred = y_pred.numpy()
        np_y = y.numpy()
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 4))
        y = torch.randint(0, 2, size=(100, 4)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_multilabel_input_NL():
    # Multilabel input data of shape (N, C, L, ...) and (N, C, L, ...)

    def _test():
        acc = Accuracy(is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(10, 4, 5))
        y = torch.randint(0, 2, size=(10, 4, 5)).long()
        acc.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, L, ...) -> (N * L ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(4, 10, 8)).long()
        y = torch.randint(0, 2, size=(4, 10, 8)).long()
        acc.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, L, ...) -> (N * L ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 4, 5))
        y = torch.randint(0, 2, size=(100, 4, 5)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, L, ...) -> (N * L ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_multilabel_input_NHW():
    # Multilabel input data of shape (N, C, H, W, ...) and (N, C, H, W, ...)

    def _test():
        acc = Accuracy(is_multilabel=True)

        y_pred = torch.randint(0, 2, size=(4, 5, 12, 10))
        y = torch.randint(0, 2, size=(4, 5, 12, 10)).long()
        acc.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, H, W, ...) -> (N * H * W ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        acc.reset()
        y_pred = torch.randint(0, 2, size=(4, 10, 12, 8)).long()
        y = torch.randint(0, 2, size=(4, 10, 12, 8)).long()
        acc.update((y_pred, y))
        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, H, W, ...) -> (N * H * W ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

        # Batched Updates
        acc.reset()
        y_pred = torch.randint(0, 2, size=(100, 5, 12, 10))
        y = torch.randint(0, 2, size=(100, 5, 12, 10)).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        np_y_pred = to_numpy_multilabel(y_pred)  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y)  # (N, C, L, ...) -> (N * L ..., C)
        assert acc._type == 'multilabel'
        assert isinstance(acc.compute(), float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(10):
        _test()


def test_incorrect_type():
    acc = Accuracy()

    # Start as binary data
    y_pred = torch.randint(0, 2, size=(4,))
    y = torch.ones(4).long()
    acc.update((y_pred, y))

    # And add a multiclass data
    y_pred = torch.rand(4, 4)
    y = torch.ones(4).long()

    with pytest.raises(RuntimeError):
        acc.update((y_pred, y))


def _test_distrib_multilabel_input_NHW(device):
    # Multilabel input data of shape (N, C, H, W, ...) and (N, C, H, W, ...)

    import torch.distributed as dist
    rank = dist.get_rank()

    def _gather(y):
        output = [torch.zeros_like(y) for i in range(dist.get_world_size())]
        dist.all_gather(output, y)
        y = torch.cat(output, dim=0)
        return y

    def _test():
        acc = Accuracy(is_multilabel=True, device=device)

        torch.manual_seed(10 + rank)
        y_pred = torch.randint(0, 2, size=(4, 5, 8, 10), device=device).long()
        y = torch.randint(0, 2, size=(4, 5, 8, 10), device=device).long()
        acc.update((y_pred, y))

        # gather y_pred, y
        y_pred = _gather(y_pred)
        y = _gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        assert acc._type == 'multilabel'
        n = acc._num_examples
        res = acc.compute()
        assert n * dist.get_world_size() == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)

        acc.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.randint(0, 2, size=(4, 7, 10, 8), device=device).long()
        y = torch.randint(0, 2, size=(4, 7, 10, 8), device=device).long()
        acc.update((y_pred, y))

        # gather y_pred, y
        y_pred = _gather(y_pred)
        y = _gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)

        assert acc._type == 'multilabel'
        n = acc._num_examples
        res = acc.compute()
        assert n * dist.get_world_size() == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)
        # check that result is not changed
        res = acc.compute()
        assert n * dist.get_world_size() == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)

        # Batched Updates
        acc.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.randint(0, 2, size=(80, 5, 8, 10), device=device).long()
        y = torch.randint(0, 2, size=(80, 5, 8, 10), device=device).long()

        batch_size = 16
        n_iters = y.shape[0] // batch_size + 1

        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx: idx + batch_size], y[idx: idx + batch_size]))

        # gather y_pred, y
        y_pred = _gather(y_pred)
        y = _gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, L, ...) -> (N * L ..., C)

        assert acc._type == 'multilabel'
        n = acc._num_examples
        res = acc.compute()
        assert n * dist.get_world_size() == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(5):
        _test()


def _test_distrib_itegration_multiclass(device):

    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs):
        n_iters = 80
        s = 16
        n_classes = 10

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * dist.get_world_size(), )).to(device)
        y_preds = torch.rand(offset * dist.get_world_size(), n_classes).to(device)

        def update(engine, i):
            return y_preds[i * s + rank * offset:(i + 1) * s + rank * offset, :], \
                y_true[i * s + rank * offset:(i + 1) * s + rank * offset]

        engine = Engine(update)

        acc = Accuracy(device=device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "acc" in engine.state.metrics
        res = engine.state.metrics['acc']
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy())

        assert pytest.approx(res) == true_res

    for _ in range(3):
        _test(n_epochs=1)
        _test(n_epochs=2)


def _test_distrib_itegration_multilabel(device):

    import torch.distributed as dist
    from ignite.engine import Engine

    rank = dist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs):
        n_iters = 80
        s = 16
        n_classes = 10

        offset = n_iters * s
        y_true = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 8, 10)).to(device)
        y_preds = torch.randint(0, 2, size=(offset * dist.get_world_size(), n_classes, 8, 10)).to(device)

        def update(engine, i):
            return y_preds[i * s + rank * offset:(i + 1) * s + rank * offset, ...], \
                y_true[i * s + rank * offset:(i + 1) * s + rank * offset, ...]

        engine = Engine(update)

        acc = Accuracy(is_multilabel=True, device=device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "acc" in engine.state.metrics
        res = engine.state.metrics['acc']
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(to_numpy_multilabel(y_true),
                                  to_numpy_multilabel(y_preds))

        assert pytest.approx(res) == true_res

    for _ in range(3):
        _test(n_epochs=1)
        _test(n_epochs=2)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    device = "cuda:{}".format(distributed_context_single_node_nccl['local_rank'])
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif('GPU_MULTINODE_DISTRIB' not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl['local_rank'])
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_itegration_multiclass(device)
    _test_distrib_itegration_multilabel(device)
