import os

import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import MeanAbsoluteError


def test_zero_div():
    mae = MeanAbsoluteError()
    with pytest.raises(NotComputableError):
        mae.compute()


def test_compute():
    mae = MeanAbsoluteError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    mae.update((y_pred, y))
    assert isinstance(mae.compute(), float)
    assert mae.compute() == 2.0

    mae.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    mae.update((y_pred, y))
    assert isinstance(mae.compute(), float)
    assert mae.compute() == 3.0


def _test_distrib_integration(device):
    import numpy as np
    from ignite.engine import Engine

    rank = idist.get_rank()
    n_iters = 80
    s = 50
    offset = n_iters * s

    y_true = torch.arange(0, offset * idist.get_world_size(), dtype=torch.float).to(device)
    y_preds = torch.ones(offset * idist.get_world_size(), dtype=torch.float).to(device)

    def update(engine, i):
        return (
            y_preds[i * s + offset * rank : (i + 1) * s + offset * rank],
            y_true[i * s + offset * rank : (i + 1) * s + offset * rank],
        )

    engine = Engine(update)

    m = MeanAbsoluteError()
    m.attach(engine, "mae")

    data = list(range(n_iters))
    engine.run(data=data, max_epochs=1)

    assert "mae" in engine.state.metrics
    res = engine.state.metrics["mae"]

    true_res = np.mean(np.abs((y_true - y_preds).cpu().numpy()))

    assert pytest.approx(res) == true_res


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_integration(device)


@pytest.mark.distributed
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = "cpu"
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = "cuda:{}".format(distributed_context_multi_node_nccl["local_rank"])
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])

    def _test_fn(index):
        device = idist.device()
        _test_distrib_integration(device)

    xmp_executor(_test_fn, args=(), nprocs=n)
