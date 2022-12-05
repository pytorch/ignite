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
    assert precision._updated is False
    with pytest.raises(NotComputableError, match=r"Precision must have at least one example before it can be computed"):
        precision.compute()
    assert precision._updated is False


def test_average_parameter():
    with pytest.raises(ValueError, match="Argument average should be None or a boolean or one of values"):
        Precision(average=1)

    pr = Precision(average="samples")
    with pytest.raises(
        ValueError, match=r"Argument average='samples' is incompatible with binary and multiclass input data."
    ):
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long()))
    assert pr._updated is False

    pr = Precision(average="samples")
    with pytest.raises(
        ValueError, match=r"Argument average='samples' is incompatible with binary and multiclass input data."
    ):
        pr.update((torch.rand(10, 3), torch.randint(0, 3, size=(10,)).long()))
    assert pr._updated is False

    pr = Precision(average=True)
    assert pr._average == "macro"


def test_binary_wrong_inputs():
    pr = Precision()

    assert pr._updated is False
    with pytest.raises(ValueError, match=r"For binary cases, y must be comprised of 0's and 1's"):
        # y has not only 0 or 1 values
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.arange(0, 10).long()))
    assert pr._updated is False

    with pytest.raises(ValueError, match=r"For binary cases, y_pred must be comprised of 0's and 1's"):
        # y_pred values are not thresholded to 0, 1 values
        pr.update((torch.rand(10), torch.randint(0, 2, size=(10,)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10, 5, 6)).long(), torch.randint(0, 2, size=(10,)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError, match=r"y must have shape of"):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))
    assert pr._updated is False

    with pytest.warns(
        RuntimeWarning,
        match="`y` and `y_pred` should be of dtype long when entry type is binary and average!=False",
    ):
        pr = Precision(average=None)
        pr.update((torch.randint(0, 2, size=(10,)).float(), torch.randint(0, 2, size=(10,))))

    with pytest.warns(
        RuntimeWarning,
        match="`y` and `y_pred` should be of dtype long when entry type is binary and average!=False",
    ):
        pr = Precision(average=None)
        pr.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).float()))


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

    pr = Precision(average=average)
    assert pr._updated is False

    def _test(y_pred, y, batch_size):
        pr.reset()
        assert pr._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            pr.update((y_pred, y))

        np_y = y.numpy().ravel()
        np_y_pred = y_pred.numpy().ravel()

        assert pr._type == "binary"
        assert pr._updated is True
        assert isinstance(pr.compute(), torch.Tensor if not average else float)
        pr_compute = pr.compute().numpy() if not average else pr.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "binary")
        assert precision_score(
            np_y, np_y_pred, average=sk_average_parameter, labels=[0, 1], zero_division=0
        ) == pytest.approx(pr_compute)

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
    pr = Precision()
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 4), torch.randint(0, 2, size=(10,)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10, 5, 6), torch.randint(0, 5, size=(10, 5)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.rand(10), torch.randint(0, 5, size=(10, 5, 6)).long()))
    assert pr._updated is False

    pr = Precision(average=True)
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))
    assert pr._updated is True

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
    assert pr._updated is True

    pr = Precision(average=False)
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).long()))
        pr.update((torch.rand(10, 6), torch.randint(0, 5, size=(10,)).long()))
    assert pr._updated is True

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.rand(10, 5, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
        pr.update((torch.rand(10, 6, 12, 14), torch.randint(0, 5, size=(10, 12, 14)).long()))
    assert pr._updated is True

    with pytest.warns(
        RuntimeWarning,
        match="`y` should be of dtype long when entry type is multiclass",
    ):
        pr = Precision()
        pr.update((torch.rand(10, 5), torch.randint(0, 5, size=(10,)).float()))


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_multiclass_input(average):

    pr = Precision(average=average)
    assert pr._updated is False

    def _test(y_pred, y, batch_size):
        pr.reset()
        assert pr._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            pr.update((y_pred, y))

        num_classes = y_pred.shape[1]
        np_y_pred = y_pred.argmax(dim=1).numpy().ravel()
        np_y = y.numpy().ravel()

        assert pr._type == "multiclass"
        assert pr._updated is True
        assert isinstance(pr.compute(), torch.Tensor if not average else float)
        pr_compute = pr.compute().numpy() if not average else pr.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "multiclass")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            sk_compute = precision_score(np_y, np_y_pred, labels=range(0, num_classes), average=sk_average_parameter)
            assert sk_compute == pytest.approx(pr_compute)

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
    pr = Precision(is_multilabel=True)
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes
        pr.update((torch.randint(0, 2, size=(10,)), torch.randint(0, 2, size=(10,)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible y_pred
        pr.update((torch.rand(10, 5), torch.randint(0, 2, size=(10, 5)).long()))
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible y
        pr.update((torch.randint(0, 5, size=(10, 5, 6)), torch.rand(10)))
    assert pr._updated is False

    with pytest.raises(ValueError):
        # incompatible shapes between two updates
        pr.update((torch.randint(0, 2, size=(20, 5)), torch.randint(0, 2, size=(20, 5)).long()))
        pr.update((torch.randint(0, 2, size=(20, 6)), torch.randint(0, 2, size=(20, 6)).long()))
    assert pr._updated is True


def to_numpy_multilabel(y):
    # reshapes input array to (N x ..., C)
    y = y.transpose(1, 0).cpu().numpy()
    num_classes = y.shape[0]
    y = y.reshape((num_classes, -1)).transpose(1, 0)
    return y


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted", "samples"])
def test_multilabel_input(average):

    pr = Precision(average=average, is_multilabel=True)
    assert pr._updated is False

    def _test(y_pred, y, batch_size):
        pr.reset()
        assert pr._updated is False

        if batch_size > 1:
            n_iters = y.shape[0] // batch_size + 1
            for i in range(n_iters):
                idx = i * batch_size
                pr.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
        else:
            pr.update((y_pred, y))

        np_y_pred = to_numpy_multilabel(y_pred)
        np_y = to_numpy_multilabel(y)

        assert pr._type == "multilabel"
        assert pr._updated is True
        pr_compute = pr.compute().numpy() if not average else pr.compute()
        sk_average_parameter = ignite_average_to_scikit_average(average, "multilabel")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y, np_y_pred, average=sk_average_parameter) == pytest.approx(pr_compute)

    def get_test_cases():

        test_cases = [
            # Multilabel input data of shape (N, C)
            (torch.randint(0, 2, size=(10, 5)), torch.randint(0, 2, size=(10, 5)), 1),
            (torch.randint(0, 2, size=(10, 4)), torch.randint(0, 2, size=(10, 4)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5)), torch.randint(0, 2, size=(50, 5)), 16),
            (torch.randint(0, 2, size=(50, 4)), torch.randint(0, 2, size=(50, 4)), 16),
            # Multilabel input data of shape (N, C, L)
            (torch.randint(0, 2, size=(10, 5, 10)), torch.randint(0, 2, size=(10, 5, 10)), 1),
            (torch.randint(0, 2, size=(10, 4, 10)), torch.randint(0, 2, size=(10, 4, 10)), 1),
            # updated batches
            (torch.randint(0, 2, size=(50, 5, 10)), torch.randint(0, 2, size=(50, 5, 10)), 16),
            (torch.randint(0, 2, size=(50, 4, 10)), torch.randint(0, 2, size=(50, 4, 10)), 16),
            # Multilabel input data of shape (N, C, H, W)
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

    pr = Precision(average=average)
    assert pr._updated is False

    y_pred = torch.softmax(torch.rand(4, 4), dim=1)
    y = torch.ones(4).long()
    pr.update((y_pred, y))
    assert pr._updated is True

    y_pred = torch.randint(0, 2, size=(4,))
    y = torch.ones(4).long()

    with pytest.raises(RuntimeError):
        pr.update((y_pred, y))

    assert pr._updated is True


@pytest.mark.parametrize("average", [None, False, "macro", "micro", "weighted"])
def test_incorrect_y_classes(average):
    pr = Precision(average=average)

    assert pr._updated is False

    y_pred = torch.randint(0, 2, size=(10, 4)).float()
    y = torch.randint(4, 5, size=(10,)).long()

    with pytest.raises(ValueError):
        pr.update((y_pred, y))

    assert pr._updated is False


def test_distrib_integration_multiclass(distributed):
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
        assert pr._updated is False

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "pr" in engine.state.metrics
        assert pr._updated is True
        res = engine.state.metrics["pr"]
        if isinstance(res, torch.Tensor):
            # Fixes https://github.com/pytorch/ignite/issues/1635#issuecomment-863026919
            assert res.device.type == "cpu"
            res = res.cpu().numpy()

        sk_average_parameter = ignite_average_to_scikit_average(average, "multiclass")
        true_res = precision_score(
            y_true.cpu().numpy(), torch.argmax(y_preds, dim=1).cpu().numpy(), average=sk_average_parameter
        )

        assert pytest.approx(res) == true_res

    metric_devices = [torch.device("cpu")]
    device = idist.device()
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
        for metric_device in metric_devices:
            _test(average=False, n_epochs=1, metric_device=metric_device)
            _test(average=False, n_epochs=2, metric_device=metric_device)
            _test(average="macro", n_epochs=1, metric_device=metric_device)
            _test(average="macro", n_epochs=2, metric_device=metric_device)
            _test(average="weighted", n_epochs=1, metric_device=metric_device)
            _test(average="weighted", n_epochs=2, metric_device=metric_device)
            _test(average="micro", n_epochs=1, metric_device=metric_device)
            _test(average="micro", n_epochs=2, metric_device=metric_device)


def test_distrib_integration_multilabel(distributed):

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
        assert pr._updated is False

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "pr" in engine.state.metrics
        assert pr._updated is True
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
        sk_average_parameter = ignite_average_to_scikit_average(average, "multilabel")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            assert precision_score(np_y_true, np_y_preds, average=sk_average_parameter) == pytest.approx(res)

    metric_devices = ["cpu"]
    device = idist.device()
    if device.type != "xla":
        metric_devices.append(idist.device())
    for _ in range(2):
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


def test_distrib_accumulator_device(distributed):
    # Binary accuracy on input of shape (N, 1) or (N, )

    def _test(average, metric_device):
        pr = Precision(average=average, device=metric_device)
        assert pr._device == metric_device
        assert pr._updated is False
        # Since the shape of the accumulated amount isn't known before the first update
        # call, the internal variables aren't tensors on the right device yet.

        y_pred = torch.randint(0, 2, size=(10,))
        y = torch.randint(0, 2, size=(10,)).long()
        pr.update((y_pred, y))

        assert pr._updated is True

        assert (
            pr._numerator.device == metric_device
        ), f"{type(pr._numerator.device)}:{pr._numerator.device} vs {type(metric_device)}:{metric_device}"

        if average != "samples":
            # For average='samples', `_denominator` is of type `int` so it has not `device` member.
            assert (
                pr._denominator.device == metric_device
            ), f"{type(pr._denominator.device)}:{pr._denominator.device} vs {type(metric_device)}:{metric_device}"

        if average == "weighted":
            assert pr._weight.device == metric_device, f"{type(pr._weight.device)}:{pr._weight.device} vs "
            f"{type(metric_device)}:{metric_device}"

    metric_devices = [torch.device("cpu")]
    device = idist.device()
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(False, metric_device=metric_device)
        _test("macro", metric_device=metric_device)
        _test("micro", metric_device=metric_device)
        _test("weighted", metric_device=metric_device)


def test_distrib_multilabel_accumulator_device(distributed):
    # Multiclass input data of shape (N, ) and (N, C)

    def _test(average, metric_device):
        pr = Precision(is_multilabel=True, average=average, device=metric_device)

        assert pr._updated is False
        assert pr._device == metric_device

        y_pred = torch.randint(0, 2, size=(10, 4, 20, 23))
        y = torch.randint(0, 2, size=(10, 4, 20, 23)).long()
        pr.update((y_pred, y))

        assert pr._updated is True

        assert (
            pr._numerator.device == metric_device
        ), f"{type(pr._numerator.device)}:{pr._numerator.device} vs {type(metric_device)}:{metric_device}"

        if average != "samples":
            # For average='samples', `_denominator` is of type `int` so it has not `device` member.
            assert (
                pr._denominator.device == metric_device
            ), f"{type(pr._denominator.device)}:{pr._denominator.device} vs {type(metric_device)}:{metric_device}"

        if average == "weighted":
            assert pr._weight.device == metric_device, f"{type(pr._weight.device)}:{pr._weight.device} vs "
            f"{type(metric_device)}:{metric_device}"

    metric_devices = [torch.device("cpu")]
    device = idist.device()
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        _test(False, metric_device=metric_device)
        _test("macro", metric_device=metric_device)
        _test("micro", metric_device=metric_device)
        _test("weighted", metric_device=metric_device)
        _test("samples", metric_device=metric_device)
