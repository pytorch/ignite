import os

import pytest
import torch

from ignite.distributed.comp_models import has_hvd_support

if not has_hvd_support:
    pytest.skip("Skip if no Horovod package", allow_module_level=True)
else:
    from ignite.distributed.comp_models.horovod import _HorovodDistModel

import horovod.torch as hvd


@pytest.mark.distributed
def test__hvd_dist_model():
    with pytest.raises(ValueError, match=r"Backend should be one of"):
        _HorovodDistModel.create_from_backend("abc")


def _assert_model(model, true_conf):

    assert model.device() == torch.device(true_conf["device"])
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_nnodes() == true_conf["nnodes"]
    assert model.get_nproc_per_node() == true_conf["nproc_per_node"]


def _test__hvd_dist_model_create_from_backend_no_dist(backend, true_device):

    model = _HorovodDistModel.create_from_backend(backend=backend)

    assert hvd.rank() > -1

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

    # Currently, there is no way to test if hvd is shutdown.
    # Only with a reduce operation that will hang ...
    # rank = model._get_hvd_rank()
    # assert rank == -1


# def _test__hvd_dist_model_create_from_backend_dist(local_rank, rank, world_size, backend, true_device):
#     import os
#     from datetime import timedelta
#
#     timeout = timedelta(seconds=20)
#     os.environ["RANK"] = "{}".format(rank)
#
#     model = _NativeDistModel.create(backend=backend, timeout=timeout)
#
#     assert dist.is_available() and dist.is_initialized()
#     assert dist.get_backend() == backend
#
#     with pytest.raises(RuntimeError, match=r"Can not create new distributed process group if default one is"):
#         _NativeDistModel.create(backend=backend, timeout=timeout)
#
#     _assert_model(
#         model,
#         {
#             "device": true_device,
#             "local_rank": local_rank,
#             "rank": rank,
#             "world_size": world_size,
#             "node_index": 0,
#             "num_nodes": 1,
#             "ntasks_per_node": world_size,
#         },
#     )
#
#     model.finalize()
#
#     del os.environ["RANK"]


def _test__hvd_dist_model_create_from_context_no_dist(true_backend, true_device):

    assert not hvd.is_initialized()
    assert _HorovodDistModel.create_from_context() is None

    hvd.init()

    true_conf = {
        "device": true_device,
        "local_rank": 0,
        "rank": 0,
        "world_size": 1,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": 1,
    }

    model = _HorovodDistModel.create_from_context()
    assert model.backend() == true_backend
    _assert_model(model, true_conf)

    hvd.shutdown()


# def _test__native_dist_model_create_from_context_dist(local_rank, rank, world_size, true_backend, true_device):
#
#     assert _NativeDistModel.create_from_context() is None
#
#     dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=world_size, rank=rank)
#     dist.barrier()
#
#     true_conf = {
#         "device": true_device,
#         "local_rank": local_rank,
#         "rank": rank,
#         "world_size": world_size,
#         "node_index": 0,
#         "num_nodes": 1,
#         "ntasks_per_node": world_size,
#     }
#
#     _test__native_dist_model_create_from_context_env_local_rank(true_conf)
#     _test__native_dist_model_create_from_context_set_local_rank(true_conf)
#
#     dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod package")
@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Skip if has GPU")
def test__hvd_dist_model_create_no_dist():
    _test__hvd_dist_model_create_from_backend_no_dist("horovod", "cpu")
    if hasattr(hvd, "is_initialized"):
        _test__hvd_dist_model_create_from_context_no_dist("horovod", "cpu")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_no_dist_nccl(clean_env):
    _test__hvd_dist_model_create_from_backend_no_dist("horovod", "cuda:0")
    if hasattr(hvd, "is_initialized"):
        _test__hvd_dist_model_create_from_context_no_dist("horovod", "cuda:0")


# @pytest.mark.distributed
# def test__native_dist_model_create_dist_gloo(local_rank, world_size):
#     _test__native_dist_model_create_from_backend_dist(local_rank, local_rank, world_size, "gloo", "cpu")
#     _test__native_dist_model_create_from_context_dist(local_rank, local_rank, world_size, "gloo", "cpu")
#
#
# @pytest.mark.distributed
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
# def test__native_dist_model_create_dist_nccl(local_rank, world_size):
#     _test__native_dist_model_create_from_backend_dist(
#         local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank)
#     )
#     _test__native_dist_model_create_from_context_dist(
#         local_rank, local_rank, world_size, "nccl", "cuda:{}".format(local_rank)
#     )
#
#
# def _test_dist_spawn_fn(local_rank, backend, world_size, device):
#     from ignite.distributed.utils import _model
#
#     assert dist.is_available() and dist.is_initialized()
#     assert dist.get_backend() == backend
#
#     assert isinstance(_model, _NativeDistModel), "{} vs _NativeDistModel".format(type(_model))
#
#     assert _model.get_local_rank() == local_rank
#     assert _model.get_world_size() == world_size
#     if backend == "nccl":
#         assert _model.device() == torch.device("{}:{}".format(device, local_rank))
#     elif backend == "gloo":
#         assert _model.device() == torch.device(device)
#
#
# def _test__native_dist_model_spawn(backend, num_workers_per_machine, device):
#     _NativeDistModel.spawn(
#         _test_dist_spawn_fn,
#         args=(backend, num_workers_per_machine, device),
#         kwargs_dict={},
#         backend=backend,
#         num_procs_per_node=num_workers_per_machine,
#     )
#
#
# @pytest.mark.distributed
# @pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
# def test__native_dist_model_spawn_gloo():
#     _test__native_dist_model_spawn("gloo", num_workers_per_machine=4, device="cpu")
#
#
# @pytest.mark.distributed
# @pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
# def test__native_dist_model_spawn_nccl():
#     _test__native_dist_model_spawn("nccl", num_workers_per_machine=torch.cuda.device_count(), device="cuda")
