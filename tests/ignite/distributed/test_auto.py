import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler

import ignite.distributed as idist
from ignite.distributed.auto import DistributedProxySampler, auto_dataloader, auto_model, auto_optim


def _test_auto_dataloader(ws, nproc, sampler_name=None, dl_type=DataLoader):

    data = torch.rand(100, 3, 12, 12)

    if sampler_name is None:
        sampler = None
    elif sampler_name == "WeightedRandomSampler":
        sampler = WeightedRandomSampler(weights=torch.ones(100), num_samples=100)
    else:
        raise RuntimeError("Unknown sampler name: {}".format(sampler_name))

    # Test auto_dataloader
    assert idist.get_world_size() == ws
    dataloader = auto_dataloader(data, batch_size=10, num_workers=2, sampler=sampler, shuffle=sampler is None)

    assert isinstance(dataloader, dl_type)
    if hasattr(dataloader, "_loader"):
        dataloader = dataloader._loader

    assert dataloader.batch_size == 10 // ws
    assert dataloader.num_workers == (2 + nproc - 1) // nproc
    if ws < 2:
        sampler_type = RandomSampler if sampler is None else type(sampler)
        assert isinstance(dataloader.sampler, sampler_type)
    else:
        sampler_type = DistributedSampler if sampler is None else DistributedProxySampler
        assert isinstance(dataloader.sampler, sampler_type)
    if isinstance(dataloader, DataLoader):
        assert dataloader.pin_memory == "cuda" in idist.device().type


def _test_auto_model_optimizer(ws, device):
    # Test auto_model
    model = nn.Linear(10, 10)
    model = auto_model(model)
    if ws > 1 and device in ("cuda", "cpu"):
        assert isinstance(model, nn.parallel.DistributedDataParallel)
    elif device != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        assert isinstance(model, nn.parallel.DataParallel)
    else:
        assert isinstance(model, nn.Module)

    assert all([p.device.type == device for p in model.parameters()]), "{} vs {}".format(
        [p.device.type for p in model.parameters()], device
    )

    # Test auto_optim
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = auto_optim(optimizer)
    if "xla" in device:
        assert isinstance(optimizer, optim.SGD) and hasattr(optimizer, "wrapped_optimizer")
    else:
        assert isinstance(optimizer, optim.SGD) and not hasattr(optimizer, "wrapped_optimizer")


def test_auto_methods_no_dist():

    _test_auto_dataloader(1, 1)
    _test_auto_dataloader(1, 1, sampler_name="WeightedRandomSampler")

    _test_auto_model_optimizer(1, "cpu")


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_auto_methods_gloo(distributed_context_single_node_gloo):

    ws = distributed_context_single_node_gloo["world_size"]
    _test_auto_dataloader(ws=ws, nproc=ws)
    _test_auto_dataloader(ws=ws, nproc=ws, sampler_name="WeightedRandomSampler")

    _test_auto_model_optimizer(ws, "cpu")


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_auto_methods_nccl(distributed_context_single_node_nccl):

    ws = distributed_context_single_node_nccl["world_size"]
    lrank = distributed_context_single_node_nccl["local_rank"]
    _test_auto_dataloader(ws=ws, nproc=ws)
    _test_auto_dataloader(ws=ws, nproc=ws, sampler_name="WeightedRandomSampler")

    device = "cuda"
    _test_auto_model_optimizer(ws, device)


def _test_auto_methods_xla(index, ws):

    dl_type = DataLoader
    if ws > 1:

        from ignite.distributed.auto import _MpDeviceLoader

        dl_type = _MpDeviceLoader
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            dl_type = MpDeviceLoader
        except ImportError:
            pass

    _test_auto_dataloader(ws=ws, nproc=ws, dl_type=dl_type)
    _test_auto_dataloader(ws=ws, nproc=ws, sampler_name="WeightedRandomSampler", dl_type=dl_type)

    device = "xla"
    _test_auto_model_optimizer(ws, device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_auto_methods_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_auto_methods_xla, args=(n,), nprocs=n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_auto_methods_xla():
    _test_auto_methods_xla(index=0, ws=1)


def test_dist_proxy_sampler():
    import torch
    from torch.utils.data import WeightedRandomSampler

    weights = torch.ones(100)
    weights[:50] += 1
    num_samples = 100
    sampler = WeightedRandomSampler(weights, num_samples)

    num_replicas = 4
    dist_samplers = [DistributedProxySampler(sampler, num_replicas=num_replicas, rank=i) for i in range(num_replicas)]

    torch.manual_seed(0)
    true_indices = list(sampler)

    indices_per_rank = []
    for s in dist_samplers:
        s.set_epoch(0)
        indices_per_rank += list(s)

    assert set(indices_per_rank) == set(true_indices)
