from typing import Callable, Union
from unittest.mock import MagicMock

import pytest
import torch
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


@pytest.mark.parametrize("n_times", range(3))
def test_binary_input(n_times, available_device, test_data_binary):
    acc = Accuracy(device=available_device)
    assert acc._device == torch.device(available_device)

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


@pytest.mark.parametrize("n_times", range(3))
def test_multiclass_input(n_times, available_device, test_data_multiclass):
    acc = Accuracy(device=available_device)
    assert acc._device == torch.device(available_device)

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


@pytest.mark.parametrize("n_times", range(3))
def test_multilabel_input(n_times, available_device, test_data_multilabel):
    acc = Accuracy(is_multilabel=True, device=available_device)
    assert acc._device == torch.device(available_device)

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


@pytest.mark.usefixtures("distributed")
class TestDistributed:
    def test_multilabel_input_NHW(self):
        # Multilabel input data of shape (N, C, H, W, ...) and (N, C, H, W, ...)
        rank = idist.get_rank()
        torch.manual_seed(10 + rank)

        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            acc = Accuracy(is_multilabel=True, device=metric_device)

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

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration_multiclass(self, n_epochs):
        rank = idist.get_rank()
        torch.manual_seed(10 + rank)

        n_iters = 80
        batch_size = 16
        n_classes = 10

        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
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

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration_multilabel(self, n_epochs):
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 80
        batch_size = 16
        n_classes = 10

        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
            metric_device = torch.device(metric_device)

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

    def test_accumulator_device(self):
        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

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

    @pytest.mark.parametrize("n_epochs", [1, 2])
    def test_integration_list_of_tensors_or_numbers(self, n_epochs):
        rank = idist.get_rank()
        torch.manual_seed(12 + rank)

        n_iters = 80
        batch_size = 16
        n_classes = 10

        metric_devices = [torch.device("cpu")]
        device = idist.device()
        if device.type != "xla":
            metric_devices.append(device)

        for metric_device in metric_devices:
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
            assert output == self.true_output

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
