import os

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

import ignite.distributed as idist
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Engine


def test_wrong_input_shapes():
    m = R2Score()

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4, 1)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1), torch.rand(4, 1, 2)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4, 1, 2), torch.rand(4,)))

    with pytest.raises(ValueError):
        m.update((torch.rand(4,), torch.rand(4, 1, 2)))


def test_r2_score():

    size = 51
    np_y_pred = np.random.rand(size,)
    np_y = np.random.rand(size,)

    m = R2Score()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    m.update((y_pred, y))

    assert r2_score(np_y, np_y_pred) == pytest.approx(m.compute())


def test_r2_score_2():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    m = R2Score()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    m.reset()
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters):
        idx = i * batch_size
        m.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))

    assert r2_score(np_y, np_y_pred) == pytest.approx(m.compute())


def test_integration_r2_score_with_output_transform():

    np.random.seed(1)
    size = 105
    np_y_pred = np.random.rand(size, 1)
    np_y = np.random.rand(size, 1)
    np.random.shuffle(np_y)

    batch_size = 15

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    m = R2Score(output_transform=lambda x: (x[1], x[2]))
    m.attach(engine, "r2_score")

    data = list(range(size // batch_size))
    r_squared = engine.run(data, max_epochs=1).metrics["r2_score"]

    assert r2_score(np_y, np_y_pred) == pytest.approx(r_squared)


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        m = R2Score(device=metric_device)
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
        assert r2_score(np_y, np_y_pred) == pytest.approx(res)

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
