import os
import warnings

import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import recall_score

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Recall

torch.manual_seed(12)


def test_no_update():
    recall = Recall()
    assert recall._updated is False
    with pytest.raises(NotComputableError, match=r"Recall must have at least one example before it can be computed"):
        recall.compute()
    assert recall._updated is False

    recall = Recall(is_multilabel=True)
    assert recall._updated is False
    with pytest.raises(NotComputableError, match=r"Recall must have at least one example before it can be computed"):
        recall.compute()
    assert recall._updated is False


def test_average_parameter():

    re = Recall(average="samples")
    with pytest.raises(
        ValueError, match=r"Argument average='samples' is incompatible with binary and multiclass input data."
    ):
        re.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long()))
    assert re._updated is False

    re = Recall(average="samples")
    with pytest.raises(
        ValueError, match=r"Argument average='samples' is incompatible with binary and multiclass input data."
    ):
        re.update((torch.rand(10, 3), torch.randint(0, 3, size=(10,)).long()))
    assert re._updated is False

    re = Recall(average=True)
    assert re._average == "macro"


def test_binary_wrong_inputs():
    re = Recall()

    assert re._updated is False
    with pytest.raises(ValueError, match=r"For binary cases, y must be comprised of 0's and 1's"):
        # y has not only 0 or 1 values
        re.update((torch.randint(0, 2, size=(10,)), torch.arange(0, 10).long()))
    assert re._updated is False

    with pytest.raises(ValueError, match=r"For binary cases, y_pred must be comprised of 0's and 1's"):
        # y_pred values are not thresholded to 0, 1 values
        re.update((torch.rand(10, 1), torch.randint(0, 2, size=(10,)).long()))
    assert re._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10, 5)).long()))
    assert re._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10, 5, 6)), torch.randint(0, 2, size=(10,)).long()))
    assert re._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10, 5, 6)).long()))
    assert re._updated is False

    with pytest.warns(
        RuntimeWarning,
        match="`y` and `y_pred` should be of dtype long when entry type is binary and average!=False",
    ):
        re = Recall(average=None)
        re.update((torch.randint(0, 2, size=(10,)).float(), torch.randint(0, 2, size=(10,))))

    with pytest.warns(
        RuntimeWarning,
        match="`y` and `y_pred` should be of dtype long when entry type is binary and average!=False",
    ):
        re = Recall(average=None)
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).float()))


def ignite_average_to_scikit_average(average, data_type: str):
    if average in [None, "micro", "samples", "weighted", "macro"]:
        return average
    if average is False:
        if data_type == "binary":
            return "binary"
        else:
            return None
    elif average is True:
        return "macro"
    else:
        raise ValueError(f"Wrong average parameter `{average}`")


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_binary_input(average):

    re = Recall(average=average)
    assert re._updated is False

    def _test(y_pred, y, batch_size):
        re.reset()
        assert re._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                re.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            re.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        assert re._type == "binary"
        assert re._updated is True
        assert isinstance(re.compute(), torch.Tensor if not average else float)
        re_compute = re.compute().numpy() if not average else re.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "binary")
        assert recall_score(np_y, np_y_pred, average=sk_average_parameter, labels=[0, 1]) == pytest.approx(re_compute)

    def get_test_cases():

        test_cases = [
            # Binary accuracy on input of shape (N, 1) or (N, )
            (torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)), 1),
            (torch.randint(0, 2, size=(10, 1)), torch.randint(0, 2, size=(10, 1)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50,)), torch.randint(0, 2, size=(50,)), 16),
            (torch.randint(0, 2, size=(50, 1)), torch.randint(0, 2, size=(50, 1)), 16),
            # Binary accuracy on input of shape (N, L)
            (torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
            (torch.randint(0, 2, size=(10, 1, 5)), torch.randint(0, 2, size=(10, 1, 5)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5)), torch.randint(0, 2, size=(50, 5)), 16),
            (torch.randint(0, 2, size=(50, 1, 5)), torch.randint(0, 2, size=(50, 1, 5)), 16),
            # Binary accuracy on input of shape (N, H, W)
            (torch.randint(0, 2, size=(10, 12, 10)), torch.randint(0, 2, size=(10, 12, 10)), 1),
            (torch.randint(0, 2, size=(10, 1, 12, 10)), torch.randint(0, 2, size=(10, 1, 12, 10)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 12, 10)), torch.randint(0, 2, size=(50, 12, 10)), 16),
            (torch.randint(0, 2, size=(50, 1, 12, 10)), torch.randint(0, 2, size=(50, 1, 12, 10)), 16),
            # Corner case with all zeros predictions
            (torch.zeros(size=(10,), dtype=torch.long), torch.randint(0, 2, size=(10,)), 1),
            (torch.zeros(size=(10, 1), dtype=torch.long), torch.randint(0, 2, size=(10, 1)), 1),
        ]

        return test_cases

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


def test_multiclass_wrong_inputs():
    re = Recall()
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).long()))
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).long()))
    assert re._updated is False

    re = Recall(average=True)
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))
    assert re._updated is True

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
    assert re._updated is True

    re = Recall(average=False)
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        re.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))
    assert re._updated is True

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        re.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
    assert re._updated is True

    with pytest.warns(
        RuntimeWarning,
        match="`y` should be of dtype long when entry type is multiclass",
    ):
        re = Recall()
        re.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).float()))


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_multiclass_input(average):

    re = Recall(average=average)
    assert re._updated is False

    def _test(y_pred, y, batch_size):
        re.reset()
        assert re._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                re.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            re.update((y_pred, y))

        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()

        assert re._type == "multiclass"
        assert re._updated is True
        assert isinstance(re.compute(), torch.Tensor if not average else float)
        re_compute = re.compute().numpy() if not average else re.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "multiclass")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = recall_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(re_compute)

    def get_test_cases():

        test_cases = [
            # Multiclass input data of shape (N, ) and (N, C)
            (torch.rand(10, 6), torch.randint(0, 6, size=(10,)), 1),
            (torch.rand(10, 4), torch.randint(0, 4, size=(10,)), 1),
            # updated batches
            (torch.rand(50, 6), torch.randint(0, 6, size=(50,)), 16),
            (torch.rand(50, 4), torch.randint(0, 4, size=(50,)), 16),
            # Multiclass input data of shape (N, L) and (N, C, L)
            (torch.rand(10, 5, 8), torch.randint(0, 5, size=(10, 8)), 1),
            (torch.rand(10, 8, 12), torch.randint(0, 8, size=(10, 12)), 1),
            # updated batches
            (torch.rand(50, 5, 8), torch.randint(0, 5, size=(50, 8)), 16),
            (torch.rand(50, 8, 12), torch.randint(0, 8, size=(50, 12)), 16),
            # Multiclass input data of shape (N, H, W, ...) and (N, C, H, W, ...)
            (torch.rand(10, 5, 18, 16), torch.randint(0, 5, size=(10, 18, 16)), 1),
            (torch.rand(10, 7, 20, 12), torch.randint(0, 7, size=(10, 20, 12)), 1),
            # updated batches
            (torch.rand(50, 5, 18, 16), torch.randint(0, 5, size=(50, 18, 16)), 16),
            (torch.rand(50, 7, 20, 12), torch.randint(0, 7, size=(50, 20, 12)), 16),
            # Corner case with all zeros predictions
            (torch.zeros(size=(10, 6)), torch.randint(0, 6, size=(10,)), 1),
            (torch.zeros(size=(10, 4)), torch.randint(0, 4, size=(10,)), 1),
        ]

        return test_cases

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


def test_multilabel_wrong_inputs():
    re = Recall(is_multilabel=True)
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        re.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).long()))
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible y_pred
        re.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).long()))
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible y
        re.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))
    assert re._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        re.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).long()))
        re.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).long()))
    assert re._updated is True


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "samples"])
def test_multilabel_input(average):

    re = Recall(average=average, is_multilabel=True)
    assert re._updated is False

    def _test(y_pred, y, batch_size):
        re.reset()
        assert re._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                re.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            re.update((y_pred, y))

        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)

        assert re._type == "multilabel"
        assert re._updated is True
        re_compute = re.compute().numpy() if not average else re.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "multilabel")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y, np_y_pred, average=sk_average_parameter) == pytest.approx(re_compute)

    def get_test_cases():

        test_cases = [
            # Multilabel input data of shape (N, C)
            (torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
            (torch.randint(0, 2, size=(10, 4)), torch.randint(0, 2, size=(10, 4)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5)), torch.randint(0, 2, size=(50, 5)), 16),
            (torch.randint(0, 2, size=(50, 4)), torch.randint(0, 2, size=(50, 4)), 16),
            # Multilabel input data of shape (N, H, W)
            (torch.randint(0, 2, size=(10, 5, 10)), torch.randint(0, 2, size=(10, 5, 10)), 1),
            (torch.randint(0, 2, size=(10, 4, 10)), torch.randint(0, 2, size=(10, 4, 10)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5, 10)), torch.randint(0, 2, size=(50, 5, 10)), 16),
            (torch.randint(0, 2, size=(50, 4, 10)), torch.randint(0, 2, size=(50, 4, 10)), 16),
            # Multilabel input data of shape (N, C, H, W, ...)
            (torch.randint(0, 2, size=(10, 5, 18, 16)), torch.randint(0, 2, size=(10, 5, 18, 16)), 1),
            (torch.randint(0, 2, size=(10, 4, 20, 23)), torch.randint(0, 2, size=(10, 4, 20, 23)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5, 18, 16)), torch.randint(0, 2, size=(50, 5, 18, 16)), 16),
            (torch.randint(0, 2, size=(50, 4, 20, 23)), torch.randint(0, 2, size=(50, 4, 20, 23)), 16),
            # Corner case with all zeros predictions
            (torch.zeros(size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
            (torch.zeros(size=(10, 4)), torch.randint(0, 2, size=(10, 4)), 1),
        ]

        return test_cases

    for _ in range(5):
        # check multiple random inputs as random exact occurencies are rare
        test_cases = get_test_cases()
        for y_pred, y, batch_size in test_cases:
            _test(y_pred, y, batch_size)


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_incorrect_type(average):
    # Tests changing of type during training

    re = Recall(average=average)
    assert re._updated is False

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).long()
    re.update((y_pred, y))
    assert re._updated is True

    y_pred = torch.zeros(4)
    y = torch.ones(4).long()

    with pytest.raises(RuntimeError):
        re.update((y_pred, y))

    assert re._updated is True


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_incorrect_y_classes(average):
    re = Recall(average=average)

    assert re._updated is False

    y_pred = torch.randint(0, 2, size=(10, 4)).float()
    y = torch.randint(4, 5, size=(10,)).long()

    with pytest.raises(ValueError):
        re.update((y_pred, y))

    assert re._updated is False


def _test_distrib_integration_multiclass(device):

    from ignite.engine import Engine

    def _test(average, n_epochs, metric_device):
        n_iters = 60
        batch_size = 16
        n_classes = 7

        y_true = torch.randint(0, n_classes, size=(n_iters * batch_size,)).to(device)
        y_preds = torch.rand(n_iters * batch_size, n_classes).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, :],
                y_true[i * batch_size : (i + 1) * batch_size],
            )

        engine = Engine(update)

        re = Recall(average=average, device=metric_device)
        re.attach(engine, "re")
        assert re._updated is False

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "re" in engine.state.metrics
        assert re._updated is True
        res = engine.state.metrics["re"]
        if isinstance(res, torch.Tensor):
            # Fixes https://github.com/pytorch/ignite/issues/1635#issuecomment-863026919
            assert res.device.type == "cpu"
            res = res.cpu().numpy()

        sk_average_parameter = ignite_average_to_scikit_average(average, "multiclass")
        true_res = recall_score(
            y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy(), average=sk_average_parameter
        )

        assert pytest.approx(res) == true_res

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    rank = idist.get_rank()
    for i in range(2):
        torch.manual_seed(12 + rank + i)
        for metric_device in metric_devices:
            _test(average=False, n_epochs=1, metric_device=metric_device)
            _test(average=False, n_epochs=2, metric_device=metric_device)
            _test(average="macro", n_epochs=1, metric_device=metric_device)
            _test(average="macro", n_epochs=2, metric_device=metric_device)
            _test(average="weighted", n_epochs=1, metric_device=metric_device)
            _test(average="weighted", n_epochs=2, metric_device=metric_device)
            _test(average="micro", n_epochs=1, metric_device=metric_device)
            _test(average="micro", n_epochs=2, metric_device=metric_device)


def _test_distrib_integration_multilabel(device):

    from ignite.engine import Engine

    torch.manual_seed(12)

    def _test(average, n_epochs, metric_device):
        n_iters = 60
        batch_size = 16
        n_classes = 7

        y_true = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 6, 8)).to(device)
        y_preds = torch.randint(0, 2, size=(n_iters * batch_size, n_classes, 6, 8)).to(device)

        def update(engine, i):
            return (
                y_preds[i * batch_size : (i + 1) * batch_size, ...],
                y_true[i * batch_size : (i + 1) * batch_size, ...],
            )

        engine = Engine(update)

        re = Recall(average=average, is_multilabel=True, device=metric_device)
        re.attach(engine, "re")
        assert re._updated is False

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        y_preds = idist.all_gather(y_preds)
        y_true = idist.all_gather(y_true)

        assert "re" in engine.state.metrics
        assert re._updated is True
        res = engine.state.metrics["re"]
        res2 = re.compute()
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()
            res2 = res2.cpu().numpy()
            assert (res == res2).all()
        else:
            assert res == res2

        np_y_preds = to_numpy_multilabel(y_preds)
        np_y_true = to_numpy_multilabel(y_true)
        assert re._type == "multilabel"
        sk_average_parameter = ignite_average_to_scikit_average(average, "multilabel")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert recall_score(np_y_true, np_y_preds, average=sk_average_parameter) == pytest.approx(res)

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    rank = idist.get_rank()
    for i in range(2):
        torch.manual_seed(12 + rank + i)
        for metric_device in metric_devices:
            _test(average=False, n_epochs=1, metric_device=metric_device)
            _test(average=False, n_epochs=2, metric_device=metric_device)
            _test(average="macro", n_epochs=1, metric_device=metric_device)
            _test(average="macro", n_epochs=2, metric_device=metric_device)
            _test(average="micro", n_epochs=1, metric_device=metric_device)
            _test(average="micro", n_epochs=2, metric_device=metric_device)
            _test(average="weighted", n_epochs=1, metric_device=metric_device)
            _test(average="weighted", n_epochs=2, metric_device=metric_device)
            _test(average="samples", n_epochs=1, metric_device=metric_device)
            _test(average="samples", n_epochs=2, metric_device=metric_device)


def _test_distrib_accumulator_device(device):
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average, metric_device):
        re = Recall(average=average, device=metric_device)
        assert re._device == metric_device
        assert re._updated is False
        # Since the shape of the accumulated amount isn't known before the first update
        # call, the internal variables aren't tensors on the right device yet.

        y_reed = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        re.update((y_reed, y))

        assert re._updated is True

        assert (
            re._numerator.device == metric_device
        ), f"{type(re._numerator.device)}:{re._numerator.device} vs {type(metric_device)}:{metric_device}"

        if average != "samples":
            # For average='samples', `_denominator` is of type `int` so it has not `device` member.
            assert (
                re._denominator.device == metric_device
            ), f"{type(re._denominator.device)}:{re._denominator.device} vs {type(metric_device)}:{metric_device}"

        if average == "weighted":
            assert re._weight.device == metric_device, f"{type(re._weight.device)}:{re._weight.device} vs "
            f"{type(metric_device)}:{metric_device}"

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(False, metric_device=metric_device)
        _test("macro", metric_device=metric_device)
        _test("micro", metric_device=metric_device)
        _test("weighted", metric_device=metric_device)


def _test_distrib_multilabel_accumulator_device(device):
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average, metric_device):
        re = Recall(is_multilabel=True, average=average, device=metric_device)

        assert re._updated is False
        assert re._device == metric_device

        y_reed = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        re.update((y_reed, y))

        assert re._updated is True

        assert (
            re._numerator.device == metric_device
        ), f"{type(re._numerator.device)}:{re._numerator.device} vs {type(metric_device)}:{metric_device}"

        if average != "samples":
            # For average='samples', `_denominator` is of type `int` so it has not `device` member.
            assert (
                re._denominator.device == metric_device
            ), f"{type(re._denominator.device)}:{re._denominator.device} vs {type(metric_device)}:{metric_device}"

        if average == "weighted":
            assert re._weight.device == metric_device, f"{type(re._weight.device)}:{re._weight.device} vs "
            f"{type(metric_device)}:{metric_device}"

    metric_devices = [torch.device("cpu")]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(False, metric_device=metric_device)
        _test("macro", metric_device=metric_device)
        _test("micro", metric_device=metric_device)
        _test("weighted", metric_device=metric_device)
        _test("samples", metric_device=metric_device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
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
    gloo_hvd_executor(_test_distrib_multilabel_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_integration_multiclass(device)
    _test_distrib_integration_multilabel(device)
    _test_distrib_accumulator_device(device)
    _test_distrib_multilabel_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
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
