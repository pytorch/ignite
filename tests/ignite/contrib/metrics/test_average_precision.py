import os

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

import ignite.distributed as idist
from ignite.contrib.metrics import AveragePrecision
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


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


def test_check_shape():
    ap = AveragePrecision()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ap._check_shape((torch.tensor(0), torch.tensor(0)))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ap._check_shape((torch.rand(4, 3, 1), torch.rand(4, 3)))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        ap._check_shape((torch.rand(4, 3), torch.rand(4, 3, 1)))


def test_average_precision():

    size = 100
    np_y_pred = np.random.rand(size, 5)
    np_y = np.random.randint(0, 2, size=(size, 5), dtype=np.long)
    np_ap = average_precision_score(np_y, np_y_pred)

    ap_metric = AveragePrecision()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    ap_metric.update((y_pred, y))
    ap = ap_metric.compute()

    assert ap == pytest.approx(np_ap)


def test_average_precision_2():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)
    np_ap = average_precision_score(np_y, np_y_pred)

    ap_metric = AveragePrecision()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ap_metric.reset()
    n_iters = 10
    batch_size = size // n_iters
    for i in range(n_iters):
        idx = i * batch_size
        ap_metric.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    ap = ap_metric.compute()

    assert ap == pytest.approx(np_ap)


def test_integration_ap_score_with_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_ap = average_precision_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ap_metric = AveragePrecision(output_transform=lambda x: (x[1], x[2]))
    ap_metric.attach(engine, "ap")

    data = list(range(size // batch_size))
    ap = engine.run(data, max_epochs=1).metrics["ap"]

    assert ap == pytest.approx(np_ap)


def test_integration_ap_score_with_activated_output_transform():

    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_softmax = torch.softmax(torch.from_numpy(np_y_pred), dim=1).numpy()
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    np_ap = average_precision_score(np_y, np_y_pred_softmax)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ap_metric = AveragePrecision(output_transform=lambda x: (torch.softmax(x[1], dim=1), x[2]))
    ap_metric.attach(engine, "ap")

    data = list(range(size // batch_size))
    ap = engine.run(data, max_epochs=1).metrics["ap"]

    assert ap == pytest.approx(np_ap)


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        ap_metric = AveragePrecision(device=metric_device)

        torch.manual_seed(10 + rank)

        y_pred = torch.rand(size=(100, 1), device=device)
        y = torch.randint(0, 2, size=(100, 1), device=device)

        ap_metric.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()

        np_ap = average_precision_score(np_y, np_y_pred)

        res = ap_metric.compute()
        assert res == pytest.approx(np_ap)

    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{distributed_context_single_node_nccl['local_rank']}")
    _test_distrib_compute(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):

    device = torch.device("cpu")
    _test_distrib_compute(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_compute, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_compute(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_compute(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
