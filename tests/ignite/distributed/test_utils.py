import os
import pytest

import torch
import torch.distributed as dist
import ignite.distributed as idist


def _sanity_check():
    from ignite.distributed.utils import _model

    assert _model.get_world_size() == _model.get_num_nodes() * _model.get_ntasks_per_node()
    assert _model.get_local_rank() < _model.get_ntasks_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node_rank() < _model.get_num_nodes()


def test_no_distrib():
    assert idist.backend() is None
    assert idist.device() == "cpu"
    assert idist.get_rank() == 0
    assert idist.get_world_size() == 1
    assert idist.get_local_rank() == 0
    assert idist.model_name() == "serial"

    from ignite.distributed.utils import _model, _SerialModel

    _sanity_check()
    assert isinstance(_model, _SerialModel)


def test__sync_model():
    pass


def _test_native_distrib(local_rank, backend, ws, device, rank=None):
    assert idist.backend() == backend, "{} vs {}".format(idist.backend(), backend)

    if backend == "nccl":
        d = "{}:{}".format(device, local_rank)
        assert idist.device() == d, "{} vs {}".format(idist.device(), d)
    elif backend == "gloo":
        assert idist.device() == device

    if rank is None:
        rank = dist.get_rank()

    assert idist.get_rank() == rank
    assert idist.get_world_size() == ws
    assert idist.get_local_rank() == local_rank

    assert idist.model_name() == "native-dist"

    from ignite.distributed.utils import _model
    from ignite.distributed.comp_models import _NativeDistModel

    _sanity_check()
    assert isinstance(_model, _NativeDistModel)


@pytest.mark.distributed
def test_native_distrib_single_node_launch_tool_gloo(local_rank, world_size):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.initialize("gloo", timeout=timeout)
    _test_native_distrib(local_rank, "gloo", world_size, "cpu", rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_launch_tool_nccl(local_rank, world_size):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.initialize("nccl", timeout=timeout)
    _test_native_distrib(local_rank, "nccl", world_size, "cuda", rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_native_distrib_single_node_spawn_gloo():

    from datetime import timedelta

    timeout = timedelta(seconds=20)

    world_size = 4

    idist.spawn(
        "gloo", _test_native_distrib, args=("gloo", world_size, "cpu"), num_procs_per_node=world_size, timeout=timeout
    )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_spawn_nccl():

    from datetime import timedelta

    timeout = timedelta(seconds=20)

    world_size = torch.cuda.device_count()

    idist.spawn(
        "nccl", _test_native_distrib, args=("nccl", world_size, "cuda"), num_procs_per_node=world_size, timeout=timeout
    )
