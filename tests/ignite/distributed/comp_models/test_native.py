import pytest

import torch
import torch.distributed as dist

from ignite.distributed.comp_models import _DistModel


def test__dist_model():
    available_backends = _DistModel.available_backends

    if dist.is_nccl_available():
        assert "nccl" in available_backends

    if dist.is_gloo_available():
        assert "gloo" in available_backends

    if dist.is_mpi_available():
        assert "mpi" in available_backends


def test__dist_model_create_from_backend_bad_config():
    import os

    os.environ["RANK"] = "1"

    with pytest.raises(RuntimeError, match=r"PyTorch distributed configuration should define env variables"):
        _DistModel.create_from_backend(backend="gloo")

    del os.environ["RANK"]


def _assert_model(model, true_conf):

    assert model.device() == true_conf["device"]
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_num_nodes() == true_conf["num_nodes"]
    assert model.get_ntasks_per_node() == true_conf["ntasks_per_node"]

    if model.get_world_size() > 1:
        assert model.is_distributed()
    else:
        assert not model.is_distributed()


def _test__dist_model_create_from_backend_no_dist(backend, true_device):

    model = _DistModel.create_from_backend(backend=backend)

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": 1,
        },
    )

    model.finalize()


def _test__dist_model_create_from_backend_dist(local_rank, rank, world_size, backend, true_device):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    os.environ["RANK"] = "{}".format(rank)

    model = _DistModel.create_from_backend(backend=backend, timeout=timeout)

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": world_size,
        },
    )

    model.finalize()

    del os.environ["RANK"]


def _test__dist_model_create_from_context_no_dist(true_backend, true_device):

    assert _DistModel.create_from_context() is None

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=1, rank=0)
    dist.barrier()

    model = _DistModel.create_from_context()

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == true_backend

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": 1,
        },
    )

    dist.destroy_process_group()


def _test__dist_model_create_from_context_dist(local_rank, rank, world_size, true_backend, true_device):

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=world_size, rank=rank)
    dist.barrier()

    model = _DistModel.create_from_context()

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == true_backend

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": world_size,
        },
    )

    dist.destroy_process_group()


def test__dist_model_create_no_dist_gloo():
    _test__dist_model_create_from_backend_no_dist("gloo", "cpu")
    # _test__dist_model_create_from_context_no_dist("gloo", "cpu")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__dist_model_create_no_dist_nccl():
    _test__dist_model_create_from_backend_no_dist("nccl", "cuda:0")
    # _test__dist_model_create_from_context_no_dist("nccl", "cuda:0")


@pytest.mark.distributed
def test__dist_model_create_dist_gloo(local_rank, world_size):
    _test__dist_model_create_from_backend_dist(local_rank, local_rank, world_size, "gloo", "cpu")
    # _test__dist_model_create_from_context_dist(local_rank, local_rank, world_size, "gloo", "cpu")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__dist_model_create_dist_nccl(local_rank, world_size):
    _test__dist_model_create_from_backend_dist(local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank))
    # _test__dist_model_create_from_context_dist(
    #     local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank)
    # )


def _test_dist_spawn_fn(local_rank, backend, world_size, device):
    from ignite.distributed.utils import _model

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    assert isinstance(_model, _DistModel), "{} vs _DistModel".format(type(_model))

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    if backend == "nccl":
        assert _model.device() == "{}:{}".format(device, local_rank)
    elif backend == "gloo":
        assert _model.device() == device


def _test__dist_model_spawn(backend, num_workers_per_machine, device):
    _DistModel.spawn(
        _test_dist_spawn_fn,
        args=(backend, num_workers_per_machine, device),
        backend=backend,
        num_procs_per_node=num_workers_per_machine,
    )


def test__dist_model_spawn_gloo():
    _test__dist_model_spawn("gloo", num_workers_per_machine=4, device="cpu")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__dist_model_spawn_nccl():
    _test__dist_model_spawn("nccl", num_workers_per_machine=torch.cuda.device_count(), device="cuda")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__dist_model_spawn_nccl():
    _test__dist_model_spawn("nccl", num_workers_per_machine=torch.cuda.device_count(), device="cuda")
