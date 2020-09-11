import os

import pytest
import torch
import torch.distributed as dist

from ignite.distributed.comp_models import has_native_dist_support

if not has_native_dist_support:
    pytest.skip("Skip if no native dist support", allow_module_level=True)
else:
    from ignite.distributed.comp_models.native import _NativeDistModel


@pytest.mark.distributed
def test__native_dist_model():
    available_backends = _NativeDistModel.available_backends

    if dist.is_nccl_available():
        assert "nccl" in available_backends
    else:
        assert "nccl" not in available_backends

    if dist.is_gloo_available():
        assert "gloo" in available_backends
    else:
        assert "gloo" not in available_backends

    if dist.is_mpi_available():
        assert "mpi" in available_backends
    else:
        assert "mpi" not in available_backends


@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_nccl_available(), reason="Skip if nccl not available")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_nccl_but_no_gpu(mock_gpu_is_not_available):

    with pytest.raises(RuntimeError, match=r"Nccl backend is required but no cuda capable devices"):
        _NativeDistModel(backend="nccl")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_dist_model_create_from_backend_bad_config():
    import os
    from datetime import timedelta

    os.environ["RANK"] = "1"

    with pytest.raises(RuntimeError, match=r"PyTorch distributed configuration should define env variables"):
        _NativeDistModel.create_from_backend(backend="gloo", timeout=timedelta(seconds=10))

    del os.environ["RANK"]


def _assert_model(model, true_conf):

    assert model.device() == torch.device(true_conf["device"])
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_nnodes() == true_conf["nnodes"]
    assert model.get_nproc_per_node() == true_conf["nproc_per_node"]


def _test__native_dist_model_create_from_backend_no_dist(backend, true_device):
    from datetime import timedelta

    model = _NativeDistModel.create_from_backend(backend=backend, timeout=timedelta(seconds=20))

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
            "nnodes": 1,
            "nproc_per_node": 1,
        },
    )

    model.finalize()


def _test__native_dist_model_create_from_backend_dist(local_rank, rank, world_size, backend, true_device):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    os.environ["RANK"] = "{}".format(rank)

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ

    model = _NativeDistModel.create_from_backend(backend=backend, timeout=timeout)

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    with pytest.raises(RuntimeError, match=r"Can not create new distributed process group if default one is"):
        _NativeDistModel.create_from_backend(backend=backend, timeout=timeout)

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "node_index": 0,
            "nnodes": 1,
            "nproc_per_node": world_size,
        },
    )

    model.finalize()

    del os.environ["RANK"]

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ
    assert "RANK" not in os.environ


def _test__native_dist_model_create_from_context_no_local_rank():

    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]

    from ignite.distributed.comp_models.base import ComputationModel

    if ComputationModel._ext_local_rank is not None:
        ComputationModel._ext_local_rank = None

    with pytest.warns(UserWarning, match=r"Local rank information for native distributed setting will be initialized"):
        _NativeDistModel.create_from_context()


def _test__native_dist_model_create_from_context_env_local_rank(true_conf):
    import os

    remove_lrank = False
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(true_conf["local_rank"])
        remove_lrank = True

    model = _NativeDistModel.create_from_context()
    _assert_model(model, true_conf)

    if remove_lrank:
        del os.environ["LOCAL_RANK"]


def _test__native_dist_model_create_from_context_set_local_rank(true_conf):

    from ignite.distributed.comp_models.base import ComputationModel

    lrank = None
    if "LOCAL_RANK" in os.environ:
        lrank = os.environ["LOCAL_RANK"]
        del os.environ["LOCAL_RANK"]

    ComputationModel._ext_local_rank = true_conf["local_rank"]

    model = _NativeDistModel.create_from_context()
    _assert_model(model, true_conf)

    ComputationModel._ext_local_rank = None

    if lrank is not None:
        os.environ["LOCAL_RANK"] = lrank


def _test__native_dist_model_create_from_context_no_dist(true_backend, true_device):

    assert _NativeDistModel.create_from_context() is None

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=1, rank=0)
    dist.barrier()

    _test__native_dist_model_create_from_context_no_local_rank()

    true_conf = {
        "device": true_device,
        "local_rank": 0,
        "rank": 0,
        "world_size": 1,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": 1,
    }

    _test__native_dist_model_create_from_context_env_local_rank(true_conf)
    _test__native_dist_model_create_from_context_set_local_rank(true_conf)

    dist.destroy_process_group()


def _test__native_dist_model_create_from_context_dist(local_rank, rank, world_size, true_backend, true_device):

    assert _NativeDistModel.create_from_context() is None

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=world_size, rank=rank)
    dist.barrier()

    true_conf = {
        "device": true_device,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": world_size,
    }

    _test__native_dist_model_create_from_context_env_local_rank(true_conf)
    _test__native_dist_model_create_from_context_set_local_rank(true_conf)

    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
def test__native_dist_model_create_no_dist_gloo(clean_env):
    _test__native_dist_model_create_from_backend_no_dist("gloo", "cpu")
    _test__native_dist_model_create_from_context_no_dist("gloo", "cpu")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_no_dist_nccl(clean_env):
    _test__native_dist_model_create_from_backend_no_dist("nccl", "cuda:0")
    _test__native_dist_model_create_from_context_no_dist("nccl", "cuda:0")


@pytest.mark.distributed
def test__native_dist_model_create_dist_gloo(local_rank, world_size):
    _test__native_dist_model_create_from_backend_dist(local_rank, local_rank, world_size, "gloo", "cpu")
    _test__native_dist_model_create_from_context_dist(local_rank, local_rank, world_size, "gloo", "cpu")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_dist_nccl(local_rank, world_size):
    _test__native_dist_model_create_from_backend_dist(
        local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank)
    )
    _test__native_dist_model_create_from_context_dist(
        local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank)
    )


def _test_dist_spawn_fn(local_rank, backend, world_size, device):
    from ignite.distributed.utils import _model

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    assert isinstance(_model, _NativeDistModel), "{} vs _NativeDistModel".format(type(_model))

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    if backend == "nccl":
        assert _model.device() == torch.device("{}:{}".format(device, local_rank))
    elif backend == "gloo":
        assert _model.device() == torch.device(device)


def _test__native_dist_model_spawn(backend, num_workers_per_machine, device, **spawn_kwargs):
    _NativeDistModel.spawn(
        _test_dist_spawn_fn,
        args=(backend, num_workers_per_machine, device),
        kwargs_dict={},
        backend=backend,
        nproc_per_node=num_workers_per_machine,
        **spawn_kwargs,
    )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_dist_model_spawn_gloo():
    _test__native_dist_model_spawn("gloo", num_workers_per_machine=4, device="cpu")
    _test__native_dist_model_spawn("gloo", num_workers_per_machine=4, device="cpu", start_method="fork")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_spawn_nccl():
    _test__native_dist_model_spawn("nccl", num_workers_per_machine=torch.cuda.device_count(), device="cuda")
