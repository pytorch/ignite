import os

import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score

import ignite.distributed as idist
from ignite.contrib.metrics import CohenKappa
from ignite.engine import Engine
from ignite.exceptions import NotComputableError


def test_no_update():
    ck = CohenKappa()

    with pytest.raises(
        NotComputableError, match=r"EpochMetric must have at least one example before it can be computed"
    ):
        ck.compute()


def test_input_types():
    ck = CohenKappa()
    ck.reset()
    output1 = (torch.rand(4, 3), torch.randint(0, 2, size=(4, 3), dtype=torch.long))
    ck.update(output1)

    with pytest.raises(ValueError, match=r"Incoherent types between input y_pred and stored predictions"):
        ck.update((torch.randint(0, 5, size=(4, 3)), torch.randint(0, 2, size=(4, 3))))

    with pytest.raises(ValueError, match=r"Incoherent types between input y and stored targets"):
        ck.update((torch.rand(4, 3), torch.randint(0, 2, size=(4, 3)).to(torch.int32)))


def test_check_shape():
    ck = CohenKappa()

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ck._check_shape((torch.randint(0, 2, size=(10, 1, 5, 12)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"Predictions should be of shape"):
        ck._check_shape((torch.randint(0, 2, size=(10, 1, 6)).long(), torch.randint(0, 2, size=(10, 5, 6)).long()))

    with pytest.raises(ValueError, match=r"Targets should be of shape"):
        ck._check_shape((torch.randint(0, 2, size=(10, 1)).long(), torch.randint(0, 2, size=(10, 5, 2)).long()))


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohen_kappa_all_weights(weights):
    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_ck = cohen_kappa_score(np_y, np_y_pred)

    ck_metric = CohenKappa(weights=weights)
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)

    ck_metric.reset()
    ck_metric.update((y_pred, y))
    ck = ck_metric.compute()

    assert ck == pytest.approx(np_ck)


def test_cohen_kappa_wrong_weights_type():
    with pytest.raises(ValueError, match=r"Kappa Weighting type must be"):
        ck = CohenKappa(weights=7)

    with pytest.raises(ValueError, match=r"Kappa Weighting type must be"):
        ck = CohenKappa(weights="dd")


@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohen_kappa_all_weights_with_output_transform(weights):
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.randint(0, 2, size=(size, 1), dtype=np.long)
    np_y = np.zeros((size,), dtype=np.long)
    np_y[size // 2 :] = 1
    np.random.shuffle(np_y)

    ck_value_sk = cohen_kappa_score(np_y, np_y_pred)

    batch_size = 10

    def update_fn(engine, batch):
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx : idx + batch_size]
        y_pred_batch = np_y_pred[idx : idx + batch_size]
        return idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    engine = Engine(update_fn)

    ck_metric = CohenKappa(output_transform=lambda x: (x[1], x[2]), weights=weights)
    ck_metric.attach(engine, "cohen_kappa")

    data = list(range(size // batch_size))
    ck_value = engine.run(data, max_epochs=1).metrics["cohen_kappa"]

    assert ck_value == pytest.approx(ck_value_sk)


def _test_distrib_compute(device):
    rank = idist.get_rank()

    def _test(metric_device):
        metric_device = torch.device(metric_device)
        ck_metric = CohenKappa(device=metric_device)

        torch.manual_seed(10 + rank)

        y_pred = torch.randint(0, 2, size=(100, 1), device=device)
        y = torch.randint(0, 2, size=(100, 1), device=device)

        ck_metric.update((y_pred, y))

        # gather y_pred, y
        y_pred = idist.all_gather(y_pred)
        y = idist.all_gather(y)

        np_y_pred = y_pred.cpu().numpy()
        np_y = y.cpu().numpy()

        np_ck = cohen_kappa_score(np_y, np_y_pred)

        res = ck_metric.compute()
        assert res == pytest.approx(np_ck)

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
