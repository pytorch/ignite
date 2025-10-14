import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler

import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader, auto_model, auto_optim, DistributedProxySampler
from tests.ignite import is_mps_available_and_functional


class DummyDS(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class DummyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(DummyIterableDataset).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Skip if WORLD_SIZE not in env vars")
def test_auto_dataloader_warning(distributed_context_single_node_gloo):
    with pytest.warns(UserWarning, match=r"Found batch_sampler in provided kwargs"):
        auto_dataloader(
            DummyDS(), batch_sampler=BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)
        )


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Skip if WORLD_SIZE not in env vars")
def test_auto_dataloader_warning_distributed_sampler(distributed_context_single_node_gloo):
    dataset = DummyDS()
    rank = idist.get_rank()
    world_size = idist.get_world_size()
    auto_dataloader(dataset, sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank))
    if world_size > 1:
        wrong_rank = (rank + 1) % world_size
        expected_warning = f"Found distributed sampler with rank={wrong_rank}, but process rank is {rank}"
        with pytest.warns(UserWarning, match=expected_warning):
            auto_dataloader(dataset, sampler=DistributedSampler(dataset, num_replicas=world_size, rank=wrong_rank))
    expected_warning = f"Found distributed sampler with num_replicas={world_size + 1}, but world size is {world_size}"
    with pytest.warns(UserWarning, match=expected_warning):
        auto_dataloader(dataset, sampler=DistributedSampler(dataset, num_replicas=world_size + 1, rank=rank))


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_auto_dataloader_warning_tpu():
    with pytest.warns(UserWarning, match=r"Found incompatible options: xla support and pin_memory"):
        auto_dataloader(DummyDS(), pin_memory=True)


def _test_auto_dataloader(ws, nproc, batch_size, num_workers=1, sampler_name=None, dl_type=DataLoader):
    def _test(data):
        if sampler_name is None:
            sampler = None
        elif sampler_name == "WeightedRandomSampler":
            sampler = WeightedRandomSampler(weights=torch.ones(100), num_samples=100)
        elif sampler_name == "DistributedSampler":
            sampler = DistributedSampler(data, num_replicas=ws, rank=idist.get_rank())
        else:
            raise RuntimeError(f"Unknown sampler name: {sampler_name}")

        # Test auto_dataloader
        assert idist.get_world_size() == ws, f"{idist.get_world_size()} vs {ws}"

        shuffle = sampler is None if not isinstance(data, IterableDataset) else False
        dataloader = auto_dataloader(
            data, batch_size=batch_size, num_workers=num_workers, sampler=sampler, shuffle=shuffle
        )

        assert isinstance(dataloader, dl_type)
        if hasattr(dataloader, "_loader"):
            dataloader = dataloader._loader
        if ws < batch_size:
            assert dataloader.batch_size == batch_size // ws
        else:
            assert dataloader.batch_size == batch_size
        if ws <= num_workers:
            assert dataloader.num_workers == (num_workers + nproc - 1) // nproc
        else:
            assert dataloader.num_workers == num_workers

        if isinstance(data, IterableDataset):
            sampler_type = _InfiniteConstantSampler
        elif ws > 1:
            if sampler is None or isinstance(sampler, DistributedSampler):
                sampler_type = DistributedSampler
            else:
                sampler_type = DistributedProxySampler
        else:
            sampler_type = RandomSampler if sampler is None else type(sampler)

        assert isinstance(dataloader.sampler, sampler_type)

        if isinstance(dataloader, DataLoader):
            assert dataloader.pin_memory == ("cuda" in idist.device().type)

    data = torch.rand(100, 3, 12, 12)
    _test(data)
    if sampler_name is None:
        data = DummyIterableDataset(0, 100)
        _test(data)


def _test_auto_model(model, ws, device, sync_bn=False, **kwargs):
    model = auto_model(model, sync_bn=sync_bn, **kwargs)
    bnd = idist.backend()
    if ws > 1 and torch.device(device).type in ("cuda", "cpu"):
        if idist.has_native_dist_support and bnd in ("nccl", "gloo"):
            assert isinstance(model, nn.parallel.DistributedDataParallel)
            if sync_bn:
                assert any([isinstance(m, nn.SyncBatchNorm) for m in model.modules()])
            if "find_unused_parameters" in kwargs:
                assert model.find_unused_parameters == kwargs["find_unused_parameters"]
        elif idist.has_hvd_support and bnd in ("horovod",):
            assert isinstance(model, nn.Module)
    elif device != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        assert isinstance(model, nn.parallel.DataParallel)
    else:
        assert isinstance(model, nn.Module)

    assert all(
        [p.device.type == torch.device(device).type for p in model.parameters()]
    ), f"{[p.device.type for p in model.parameters()]} vs {torch.device(device).type}"


def _test_auto_model_optimizer(ws, device):
    # Test auto_model
    model = nn.Linear(10, 10)
    _test_auto_model(model, ws, device)

    model = nn.Sequential(nn.Linear(20, 100), nn.BatchNorm1d(100))
    _test_auto_model(model, ws, device, sync_bn="cuda" in torch.device(device).type)
    if ws > 1:
        _test_auto_model(model, ws, device, find_unused_parameters=True)
        _test_auto_model(model, ws, device, find_unused_parameters=False)

    # Test auto_optim
    bnd = idist.backend()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = auto_optim(optimizer)
    if idist.has_xla_support and "xla" in device:
        assert isinstance(optimizer, optim.SGD) and hasattr(optimizer, "wrapped_optimizer")
    elif idist.has_hvd_support and bnd in ("horovod",):
        assert isinstance(optimizer, optim.SGD) and hasattr(optimizer, "_allreduce_grad_async")
    else:
        assert isinstance(optimizer, optim.SGD) and not hasattr(optimizer, "wrapped_optimizer")

    if idist.has_hvd_support and bnd in ("horovod",):
        backward_passes_per_step = 2
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = auto_optim(optimizer, backward_passes_per_step=backward_passes_per_step)
        assert isinstance(optimizer, optim.SGD) and hasattr(optimizer, "backward_passes_per_step")
        assert optimizer.backward_passes_per_step == backward_passes_per_step


@pytest.mark.skipif(not is_mps_available_and_functional(), reason="Skip if MPS not functional")
def test_auto_methods_no_dist():
    _test_auto_dataloader(1, 1, batch_size=1)
    _test_auto_dataloader(1, 1, batch_size=10, num_workers=2)
    _test_auto_dataloader(1, 1, batch_size=10, sampler_name="WeightedRandomSampler")
    _test_auto_dataloader(1, 1, batch_size=10, sampler_name="DistributedSampler")
    device = idist.device()
    _test_auto_model_optimizer(1, device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_auto_methods_gloo(distributed_context_single_node_gloo):
    ws = distributed_context_single_node_gloo["world_size"]
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=10, num_workers=2)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=10, sampler_name="WeightedRandomSampler")
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=10, sampler_name="DistributedSampler")

    device = idist.device()
    _test_auto_model_optimizer(ws, device)

    if ws > 1 and device.type == "cpu":
        # Pytorch <= 1.9.0 => AssertionError
        # Pytorch >  1.9   => ValueError
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/distributed.py#L1498
        with pytest.raises(
            (AssertionError, ValueError), match=r"SyncBatchNorm layers only work with (GPU|CUDA) modules"
        ):
            model = nn.Sequential(nn.Linear(20, 100), nn.BatchNorm1d(100))
            auto_model(model, sync_bn=True)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_auto_methods_nccl(distributed_context_single_node_nccl):
    ws = distributed_context_single_node_nccl["world_size"]
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=10, num_workers=10)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1, sampler_name="WeightedRandomSampler")
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1, sampler_name="DistributedSampler")

    device = idist.device()
    _test_auto_model_optimizer(ws, device)

    if ws > 1:
        with pytest.raises(ValueError, match=r"Argument kwargs should not contain 'device_ids'"):
            auto_model(nn.Linear(1, 1), device_ids=[0])


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_auto_methods_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_auto_dataloader, args=(np, np, 1), np=np, do_init=True)
    gloo_hvd_executor(_test_auto_dataloader, args=(np, np, 10, 10), np=np, do_init=True)
    gloo_hvd_executor(_test_auto_dataloader, args=(np, np, 1, 1, "WeightedRandomSampler"), np=np, do_init=True)
    gloo_hvd_executor(_test_auto_dataloader, args=(np, np, 1, 1, "DistributedSampler"), np=np, do_init=True)

    gloo_hvd_executor(_test_auto_model_optimizer, args=(np, device), np=np, do_init=True)


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

    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1, dl_type=dl_type)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=10, num_workers=2, dl_type=dl_type)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1, sampler_name="WeightedRandomSampler", dl_type=dl_type)
    _test_auto_dataloader(ws=ws, nproc=ws, batch_size=1, sampler_name="DistributedSampler", dl_type=dl_type)

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
    weights = torch.ones(100)
    weights[:50] += 1
    num_samples = 200
    sampler = WeightedRandomSampler(weights, num_samples)

    num_replicas = 8
    dist_samplers = [DistributedProxySampler(sampler, num_replicas=num_replicas, rank=i) for i in range(num_replicas)]

    for seed in range(100):
        torch.manual_seed(seed)
        true_indices = list(sampler)

        indices_per_rank = []
        for s in dist_samplers:
            s.set_epoch(seed)
            indices_per_rank += list(s)

        set_indices_per_rank = set(indices_per_rank)
        set_true_indices = set(true_indices)
        assert (
            set_indices_per_rank == set_true_indices
        ), f"{set_true_indices - set_indices_per_rank} | {set_indices_per_rank - set_true_indices}"

    with pytest.raises(TypeError, match=r"Argument sampler should be instance of torch Sampler"):
        DistributedProxySampler(None)

    with pytest.raises(TypeError, match=r"Argument sampler should have length"):
        DistributedProxySampler(Sampler())

    with pytest.raises(TypeError, match=r"Argument sampler must not be a distributed sampler already"):
        DistributedProxySampler(DistributedSampler(sampler, num_replicas=num_replicas, rank=0))
