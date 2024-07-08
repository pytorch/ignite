import os
from typing import Callable, Union
from unittest.mock import MagicMock

import pytest
import torch
from packaging.version import Version
from sklearn.metrics import accuracy_score

import ignite.distributed as idist
from ignite.engine import Engine, State
from ignite.exceptions import NotComputableError
from ignite.metrics import Accuracy

torch.manual_seed(12)


def test_no_update():
    acc = Accuracy()
    with pytest.raises(NotComputableError, match=r"Accuracy must have at least one example before it can be computed"):
        acc.compute()


def test__check_shape():
    acc = Accuracy()

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes"):
        acc._check_shape((torch.randint(0, 2, size=(10, 1, 5, 12)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes"):
        acc._check_shape((torch.randint(0, 2, size=(10, 1, 6)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"y and y_pred must have compatible shapes"):
        acc._check_shape((torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test__check_type():
    acc = Accuracy()

    with pytest.raises(RuntimeError, match=r"Invalid shapes of y"):
        acc._check_type((torch.rand([1, 1, 1]), torch.rand([1])))


def test_binary_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError, match=r"For binary cases, y must be comprised of 0's and 1's"):
        # y has not only 0 or 1 values
        acc.update((torch.randint(0, 2, size=(10,)).long(), torch.arange(0, 10).long()))

    with pytest.raises(ValueError, match=r"For binary cases, y_pred must be comprised of 0's and 1's"):
        # y_pred values are not thresholded to 0, 1 values
        acc.update((torch.rand(10), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y must have shape of "):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))

    with pytest.raises(ValueError, match=r"y must have shape of "):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10, 5, 6)).long(), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError, match=r"y must have shape of "):
        # incompatible shapes
        acc.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))


@pytest.fixture(params=range(12))
def test_data_binary(request):
    return [
        # Binary accuracy on input of shape (N, 1) or (N, )
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50,)).long(), torch.randint(0, 2, size=(50,)).long(), 16),
        (torch.randint(0, 2, size=(50, 1)).long(), torch.randint(0, 2, size=(50, 1)).long(), 16),
        # Binary accuracy on input of shape (N, L)
        (torch.randint(0, 2, size=(10, 5)).long(), torch.randint(0, 2, size=(10, 5)).long(), 1),
        (torch.randint(0, 2, size=(10, 8)).long(), torch.randint(0, 2, size=(10, 8)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5)).long(), torch.randint(0, 2, size=(50, 5)).long(), 16),
        (torch.randint(0, 2, size=(50, 8)).long(), torch.randint(0, 2, size=(50, 8)).long(), 16),
        # Binary accuracy on input of shape (N, H, W, ...)
        (torch.randint(0, 2, size=(4, 1, 12, 10)).long(), torch.randint(0, 2, size=(4, 1, 12, 10)).long(), 1),
        (torch.randint(0, 2, size=(15, 1, 20, 10)).long(), torch.randint(0, 2, size=(15, 1, 20, 10)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 1, 12, 10)).long(), torch.randint(0, 2, size=(50, 1, 12, 10)).long(), 16),
        (torch.randint(0, 2, size=(50, 1, 20, 10)).long(), torch.randint(0, 2, size=(50, 1, 20, 10)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_binary_input(n_times, test_data_binary):
    acc = Accuracy()

    y_pred, y, batch_size = test_data_binary
    acc.reset()
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        acc.update((y_pred, y))

    np_y = y.numpy().ravel()
    np_y_pred = y_pred.numpy().ravel()

    assert acc._type == "binary"
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


def test_multiclass_wrong_inputs():
    acc = Accuracy()

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).long()))

    with pytest.raises(ValueError):
        # incompatible shapes
        acc.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).long()))


@pytest.fixture(params=range(11))
def test_data_multiclass(request):
    return [
        # Multiclass input data of shape (N, ) and (N, C)
        (torch.rand(10, 4), torch.randint(0, 4, size=(10,)).long(), 1),
        (torch.rand(10, 10, 1), torch.randint(0, 18, size=(10, 1)).long(), 1),
        (torch.rand(10, 18), torch.randint(0, 18, size=(10,)).long(), 1),
        (torch.rand(4, 10), torch.randint(0, 10, size=(4,)).long(), 1),
        # 2-classes
        (torch.rand(4, 2), torch.randint(0, 2, size=(4,)).long(), 1),
        (torch.rand(100, 5), torch.randint(0, 5, size=(100,)).long(), 16),
        # Multiclass input data of shape (N, L) and (N, C, L)
        (torch.rand(10, 4, 5), torch.randint(0, 4, size=(10, 5)).long(), 1),
        (torch.rand(4, 10, 5), torch.randint(0, 10, size=(4, 5)).long(), 1),
        (torch.rand(100, 9, 7), torch.randint(0, 9, size=(100, 7)).long(), 16),
        # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
        (torch.rand(4, 5, 12, 10), torch.randint(0, 5, size=(4, 12, 10)).long(), 1),
        (torch.rand(100, 3, 8, 8), torch.randint(0, 3, size=(100, 8, 8)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_multiclass_input(n_times, test_data_multiclass):
    acc = Accuracy()

    y_pred, y, batch_size = test_data_multiclass
    acc.reset()
    if batch_size > 1:
        # Batched Updates
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        acc.update((y_pred, y))

    np_y_pred = y_pred.numpy().argmax(axis=1).ravel()
    np_y = y.numpy().ravel()

    assert acc._type == "multiclass"
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


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

    with pytest.raises(ValueError):
        # incompatible binary shapes
        acc.update((torch.randint(0, 2, size=(10, 1)), torch.randint(0, 2, size=(10, 1)).long()))


@pytest.fixture(params=range(12))
def test_data_multilabel(request):
    return [
        # Multilabel input data of shape (N, C) and (N, C)
        (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 1),
        (torch.randint(0, 2, size=(10, 7)).long(), torch.randint(0, 2, size=(10, 7)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 4)).long(), torch.randint(0, 2, size=(50, 4)).long(), 16),
        (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 16),
        # Multilabel input data of shape (N, H, W)
        (torch.randint(0, 2, size=(10, 5, 10)).long(), torch.randint(0, 2, size=(10, 5, 10)).long(), 1),
        (torch.randint(0, 2, size=(10, 4, 10)).long(), torch.randint(0, 2, size=(10, 4, 10)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5, 10)).long(), torch.randint(0, 2, size=(50, 5, 10)).long(), 16),
        (torch.randint(0, 2, size=(50, 4, 10)).long(), torch.randint(0, 2, size=(50, 4, 10)).long(), 16),
        # Multilabel input data of shape (N, C, H, W, ...) and (N, C, H, W, ...)
        (torch.randint(0, 2, size=(4, 5, 12, 10)).long(), torch.randint(0, 2, size=(4, 5, 12, 10)).long(), 1),
        (torch.randint(0, 2, size=(4, 10, 12, 8)).long(), torch.randint(0, 2, size=(4, 10, 12, 8)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(50, 5, 12, 10)).long(), torch.randint(0, 2, size=(50, 5, 12, 10)).long(), 16),
        (torch.randint(0, 2, size=(50, 10, 12, 8)).long(), torch.randint(0, 2, size=(50, 10, 12, 8)).long(), 16),
    ][request.param]


@pytest.mark.parametrize("n_times", range(5))
def test_multilabel_input(n_times, test_data_multilabel):
    acc = Accuracy(is_multilabel=True)

    y_pred, y, batch_size = test_data_multilabel
    if batch_size > 1:
        n_iters = y.shape[0] // batch_size + 1
        for i in range(n_iters):
            idx = i * batch_size
            acc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    else:
        acc.update((y_pred, y))

    np_y_pred = to_numpy_multilabel(y_pred)
    np_y = to_numpy_multilabel(y)

    assert acc._type == "multilabel"
    assert isinstance(acc.compute(), float)
    assert accuracy_score(np_y, np_y_pred) == pytest.approx(acc.compute())


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

    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        acc = Accuracy(is_multilabel=True, device=metric_device)

        torch.manual_seed(10 + rank)
        y_pred = torch.randint(0, 2, size=(4, 5, 8, 10), device=device).long()
        y = torch.randint(0, 2, size=(4, 5, 8, 10), device=device).long()
        acc.update((y_pred, y))

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        n = acc._num_examples
        assert n == y.numel() / y.size(dim=1)

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        assert acc._type == "multilabel"
        res = acc.compute()
        assert n == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)

        acc.reset()
        torch.manual_seed(10 + rank)
        y_pred = torch.randint(0, 2, size=(4, 7, 10, 8), device=device).long()
        y = torch.randint(0, 2, size=(4, 7, 10, 8), device=device).long()
        acc.update((y_pred, y))

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        n = acc._num_examples
        assert n == y.numel() / y.size(dim=1)

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, H, W, ...) -> (N * H * W ..., C)

        assert acc._type == "multilabel"
        res = acc.compute()
        assert n == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)
        # check that result is not changed
        res = acc.compute()
        assert n == acc._num_examples
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
            acc.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        n = acc._num_examples
        assert n == y.numel() / y.size(dim=1)

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = to_numpy_multilabel(y_pred.cpu())  # (N, C, L, ...) -> (N * L * ..., C)
        np_y = to_numpy_multilabel(y.cpu())  # (N, C, L, ...) -> (N * L ..., C)

        assert acc._type == "multilabel"
        res = acc.compute()
        assert n == acc._num_examples
        assert isinstance(res, float)
        assert accuracy_score(np_y, np_y_pred) == pytest.approx(res)

    # check multiple random inputs as random exact occurencies are rare
    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


def _test_distrib_integration_multiclass(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16
        n_classes = 10

        torch.manual_seed(12 + rank)

        y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, :],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        acc = Accuracy(device=metric_device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy())

        assert pytest.approx(res) == true_res

        metric_state = acc.state_dict()
        saved__num_correct = acc._num_correct
        saved__num_examples = acc._num_examples
        acc.reset()
        acc.load_state_dict(metric_state)
        assert acc._num_examples == saved__num_examples
        assert (acc._num_correct == saved__num_correct).all()

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


def _test_distrib_integration_multilabel(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16
        n_classes = 10

        torch.manual_seed(12 + rank)

        y_true = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 8, 10)).to(device)
        y_preds = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 8, 10)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, ...],
                y_true[i * batch_size : (i + 1) * batch_size, ...],
            )

        engine = Engine(update)

        acc = Accuracy(is_multilabel=True, device=metric_device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(to_numpy_multilabel(y_true), to_numpy_multilabel(y_preds))

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


def _test_distrib_accumulator_device(device):
    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        acc = Accuracy(device=metric_device)
        assert acc._device == metric_device
        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        y_pred = torch.randint(0, 2, size=(10,), device=device, dtype=torch.long)
        y = torch.randint(0, 2, size=(10,), device=device, dtype=torch.long)
        acc.update((y_pred, y))

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"


def _test_distrib_integration_list_of_tensors_or_numbers(device):
    rank = idist.get_rank()

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        batch_size = 16
        n_classes = 10

        torch.manual_seed(12 + rank)

        y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

        def update(_, i):
            return (
                [v for v in y_preds[i * batch_size : (i + 1) * batch_size, ...]],
                [v.item() for v in y_true[i * batch_size : (i + 1) * batch_size]],
            )

        engine = Engine(update)

        acc = Accuracy(device=metric_device)
        acc.attach(engine, "acc")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_true = idist.all_gather(y_true)
        y_preds = idist.all_gather(y_preds)

        assert (
            acc._num_correct.device == metric_device
        ), f"{type(acc._num_correct.device)}:{acc._num_correct.device} vs {type(metric_device)}:{metric_device}"

        assert "acc" in engine.state.metrics
        res = engine.state.metrics["acc"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = accuracy_score(y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy())
        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="Skip if < 1.7.0")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_multilabel_input_NHW, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_multiclass, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_multilabel, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_list_of_tensors_or_numbers, (device,), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_distrib_multilabel_input_NHW(device)
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_integration_list_of_tensors_or_numbers(device)


def test_skip_unrolling():
    class DummyAcc(Accuracy):
        def __init__(
            self,
            true_output,
            output_transform: Callable = lambda x: x,
            is_multilabel: bool = False,
            device: Union[str, torch.device] = torch.device("cpu"),
            skip_unrolling: bool = False,
        ):
            super(DummyAcc, self).__init__(
                output_transform=output_transform, is_multilabel=False, device=device, skip_unrolling=skip_unrolling
            )
            self.true_output = true_output

        def update(self, output):
            assert torch.all(output == self.true_output)

    a_pred = torch.randint(0, 2, size=(8, 1))
    b_pred = torch.randint(0, 2, size=(8, 1))
    y_pred = [a_pred, b_pred]
    a_true = torch.randint(0, 2, size=(8, 1))
    b_true = torch.randint(0, 2, size=(8, 1))
    y_true = [a_true, b_true]

    acc = DummyAcc(true_output=(y_pred, y_true), skip_unrolling=True)
    state = State(output=(y_pred, y_true))
    engine = MagicMock(state=state)
    acc.iteration_completed(engine)
