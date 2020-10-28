import os

import numpy as np
import pytest
import torch
from sklearn.neighbors import DistanceMetric

import ignite.distributed as idist
from ignite.contrib.metrics.regression import CanberraMetric


def test_wrong_input_shapes():
    m = CanberraMetric()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 1, 2)))


def test_compute():
    a = np.random.randn(4)
    b = np.random.randn(4)
    c = np.random.randn(4)
    d = np.random.randn(4)
    ground_truth = np.random.randn(4)

    m = CanberraMetric()

    canberra = DistanceMetric.get_metric("canberra")

    m.update((torch.from_numpy(a), torch.from_numpy(ground_truth)))
    np_sum = (np.abs(ground_truth - a) / (np.abs(a) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    assert canberra.pairwise([a, ground_truth])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(b), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - b)) / (np.abs(b) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([a, b])
    v2 = np.hstack([ground_truth, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(c), torch.from_numpy(ground_truth)))
    np_sum += ((np.abs(ground_truth - c)) / (np.abs(c) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([v1, c])
    v2 = np.hstack([v2, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)

    m.update((torch.from_numpy(d), torch.from_numpy(ground_truth)))
    np_sum += (np.abs(ground_truth - d) / (np.abs(d) + np.abs(ground_truth))).sum()
    assert m.compute() == pytest.approx(np_sum)
    v1 = np.hstack([v1, d])
    v2 = np.hstack([v2, ground_truth])
    assert canberra.pairwise([v1, v2])[0][1] == pytest.approx(np_sum)


def _test_distrib_compute(device, tol=1e-4):
    rank = idist.get_rank()

    canberra = DistanceMetric.get_metric("canberra")

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = CanberraMetric(device=metric_device)
        torch.manual_seed(10 + rank)

        y_pred = torch.randint(0, 10, size=(10,), device=device).float()
        y = torch.randint(0, 10, size=(10,), device=device).float()

        m.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()
        res = m.compute()
        assert canberra.pairwise([np_y_pred, np_y])[0][1] == pytest.approx(res, abs=tol)

    for _ in range(3):
        _test("cpu")
        if device.type != "xla":
            _test(idist.device())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(distributed_context_single_node_nccl):
    device = torch.device("cuda:{}".format(distributed_context_single_node_nccl["local_rank"]))
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
    device = torch.device("cuda:{}".format(distributed_context_multi_node_nccl["local_rank"]))
    _test_distrib_compute(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_compute(device, tol=1.e-3)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_compute(device, tol=1.e-3)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
