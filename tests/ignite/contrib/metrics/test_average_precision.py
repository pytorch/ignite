import os

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

import ignite.distributed as idist
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Engine
from ignite.exceptions import NotComputableError

torch.manual_seed(12)


def test_no_update():
    ap = AveragePrecision()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        ap.compute()


def test_input_types():
    ap = AveragePrecision()
    ap.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    ap.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        ap.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        ap.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        ap.update((torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10, 5)).long()))


def test_check_shape():
    ap = AveragePrecision()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ap._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ap._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        ap._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


def test_binary_input_N():
    ap = AveragePrecision()

    def _test(y_pred, y, n_iters):
        ap.reset()
        ap.update((y_pred, y))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        if n_iters > 1:
            batch_size = y.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                ap.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        res = ap.compute()
        assert isinstance(res, float)
        assert average_precision_score(np_y, np_y_pred) == pytest.approx(res)

    test_cases = [
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 1),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16),
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 16),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 16),
    ]

    for y_pred, y, n_iters in test_cases:
        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test(y_pred, y, n_iters)


def test_multilabel_input_N():

    ap = AveragePrecision()

    def _test(y_pred, y, n_iters):
        ap.reset()
        ap.update((y_pred, y))

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        if n_iters > 1:
            batch_size = y.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                ap.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        res = ap.compute()
        assert isinstance(res, float)
        assert average_precision_score(np_y, np_y_pred) == pytest.approx(res)

    test_cases = [
        (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 1),
        (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 1),
        (torch.randint(0, 2, size=(100, 4)).long(), torch.randint(0, 2, size=(100, 4)).long(), 1),
        (torch.randint(0, 2, size=(200, 6)).long(), torch.randint(0, 2, size=(200, 6)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 16),
        (torch.randint(0, 2, size=(50, 7)).long(), torch.randint(0, 2, size=(50, 7)).long(), 16),
        (torch.randint(0, 2, size=(100, 4)).long(), torch.randint(0, 2, size=(100, 4)).long(), 16),
        (torch.randint(0, 2, size=(200, 6)).long(), torch.randint(0, 2, size=(200, 6)).long(), 16),
    ]

    for y_pred, y, n_iters in test_cases:
        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test(y_pred, y, n_iters)


def test_integration_binary_input_with_output_transform():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        ap_metric = AveragePrecision(output_transform=lambda x: (x[1], x[2]))
        ap_metric.attach(engine, "ap")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        np_ap = average_precision_score(np_y, np_y_pred)

        data = list(range(y_pred.shape[0] // batch_size))
        ap = engine.run(data, max_epochs=1).metrics["ap"]

        assert isinstance(ap, float)
        assert np_ap == pytest.approx(ap)

    test_cases = [
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 10),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 10),
        (torch.randint(0, 2, size=(200,)).long(), torch.randint(0, 2, size=(200,)).long(), 10),
        (torch.randint(0, 2, size=(200, 1)).long(), torch.randint(0, 2, size=(200, 1)).long(), 10),
    ]

    for y_pred, y, batch_size in test_cases:
        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test(y_pred, y, batch_size)


def test_integration_multilabel_input_with_output_transform():
    def _test(y_pred, y, batch_size):
        def update_fn(engine, batch):
            idx = (engine.state.iteration - 1) * batch_size
            y_true_batch = np_y[idx : idx + batch_size]
            y_pred_batch = np_y_pred[idx : idx + batch_size]
            return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

        engine = Engine(update_fn)

        ap_metric = AveragePrecision(output_transform=lambda x: (x[1], x[2]))
        ap_metric.attach(engine, "ap")

        np_y = y.numpy()
        np_y_pred = y_pred.numpy()

        np_ap = average_precision_score(np_y, np_y_pred)

        data = list(range(y_pred.shape[0] // batch_size))
        ap = engine.run(data, max_epochs=1).metrics["ap"]

        assert isinstance(ap, float)
        assert np_ap == pytest.approx(ap)

    test_cases = [
        (torch.randint(0, 2, size=(100, 3)).long(), torch.randint(0, 2, size=(100, 3)).long(), 10),
        (torch.randint(0, 2, size=(100, 4)).long(), torch.randint(0, 2, size=(100, 4)).long(), 10),
        (torch.randint(0, 2, size=(200, 5)).long(), torch.randint(0, 2, size=(200, 5)).long(), 10),
        (torch.randint(0, 2, size=(200, 6)).long(), torch.randint(0, 2, size=(200, 6)).long(), 10),
    ]

    for y_pred, y, batch_size in test_cases:
        # check multiple random inputs as random exact occurencies are rare
        for _ in range(10):
            _test(y_pred, y, batch_size)


def _test_distrib_binary_input_N(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(y_pred, y, n_iters, metric_device):

        metric_device = torch.device(metric_device)
        ap = AveragePrecision(device=metric_device)
        torch.manual_seed(10 + rank)

        ap.reset()
        ap.update((y_pred, y))

        if n_iters > 1:
            batch_size = y.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                ap.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = ap.compute()
        assert isinstance(res, float)
        assert average_precision_score(np_y, np_y_pred) == pytest.approx(res)

    test_cases = [
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 1),
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 1),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 1),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(10,)).long(), torch.randint(0, 2, size=(10,)).long(), 16),
        (torch.randint(0, 2, size=(100,)).long(), torch.randint(0, 2, size=(100,)).long(), 16),
        (torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 1)).long(), 16),
        (torch.randint(0, 2, size=(100, 1)).long(), torch.randint(0, 2, size=(100, 1)).long(), 16),
    ]

    for y_pred, y, batch_size in test_cases:
        for _ in range(3):
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_multilabel_input_N(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(y_pred, y, n_iters, metric_device):

        metric_device = torch.device(metric_device)
        ap = AveragePrecision(device=metric_device)
        torch.manual_seed(10 + rank)

        ap.reset()
        ap.update((y_pred, y))

        if n_iters > 1:
            batch_size = y.shape[0] // n_iters + 1
            for i in range(n_iters):
                idx = i * batch_size
                ap.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y = y.cpu().numpy()
        np_y_pred = y_pred.cpu().numpy()

        res = ap.compute()
        assert isinstance(res, float)
        assert average_precision_score(np_y, np_y_pred) == pytest.approx(res)

    test_cases = [
        (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 1),
        (torch.randint(0, 2, size=(100, 7)).long(), torch.randint(0, 2, size=(100, 7)).long(), 1),
        (torch.randint(0, 2, size=(100, 5)).long(), torch.randint(0, 2, size=(100, 5)).long(), 1),
        (torch.randint(0, 2, size=(100, 3)).long(), torch.randint(0, 2, size=(100, 3)).long(), 1),
        # updated batches
        (torch.randint(0, 2, size=(10, 4)).long(), torch.randint(0, 2, size=(10, 4)).long(), 16),
        (torch.randint(0, 2, size=(100, 7)).long(), torch.randint(0, 2, size=(100, 7)).long(), 16),
        (torch.randint(0, 2, size=(100, 5)).long(), torch.randint(0, 2, size=(100, 5)).long(), 16),
        (torch.randint(0, 2, size=(100, 3)).long(), torch.randint(0, 2, size=(100, 3)).long(), 16),
    ]

    for y_pred, y, batch_size in test_cases:
        for _ in range(3):
            _test(y_pred, y, batch_size, "cpu")
            if device.type != "xla":
                _test(y_pred, y, batch_size, idist.device())


def _test_distrib_integration_binary(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 2

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(),)).to(device)
        y_preds = torch.rand(offset * idist.get_world_size(),).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset],
            )

        engine = Engine(update)

        ap = AveragePrecision(device=metric_device)
        ap.attach(engine, "ap")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "ap" in engine.state.metrics

        res = engine.state.metrics["ap"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = average_precision_score(y_true.cpu().numpy(), y_preds.cpu().numpy())

        assert pytest.approx(res) == true_res

    metric_devices = ["cpu"]
    if device.type != "xla":
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        for _ in range(2):
            _test(n_epochs=1, metric_device=metric_device)
            _test(n_epochs=2, metric_device=metric_device)


def _test_distrib_integration_multilabel(device):

    rank = idist.get_rank()
    torch.manual_seed(12)

    def _test(n_epochs, metric_device):
        metric_device = torch.device(metric_device)
        n_iters = 80
        s = 16
        n_classes = 2

        offset = n_iters * s
        y_true = torch.randint(0, n_classes, size=(offset * idist.get_world_size(), 10)).to(device)
        y_preds = torch.randint(0, n_classes, size=(offset * idist.get_world_size(), 10)).to(device)

        def update(engine, i):
            return (
                y_preds[i * s + rank * offset : (i + 1) * s + rank * offset, :],
                y_true[i * s + rank * offset : (i + 1) * s + rank * offset, :],
            )

        engine = Engine(update)

        ap = AveragePrecision(device=metric_device)
        ap.attach(engine, "ap")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=n_epochs)

        assert "ap" in engine.state.metrics

        res = engine.state.metrics["ap"]
        if isinstance(res, torch.Tensor):
            res = res.cpu().numpy()

        true_res = average_precision_score(y_true.cpu().numpy(), y_preds.cpu().numpy())

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
def test_distrib_gpu(distributed_context_single_node_nccl):

    device = torch.device(f"cuda:{distributed_context_single_node_nccl['local_rank']}")
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_binary_input_N, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_multilabel_input_N, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_binary, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_integration_multilabel, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):

    device = torch.device("cpu")
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):

    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():

    device = idist.device()
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_binary_input_N(device)
    _test_distrib_multilabel_input_N(device)
    _test_distrib_integration_binary(device)
    _test_distrib_integration_multilabel(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
