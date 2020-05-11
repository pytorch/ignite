import os

import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import has_xla_support


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
    elif backend == "xla-tpu":
        d = idist.device()
        assert isinstance(d, torch.device) and device in d.type

    if rank is None:
        if idist.model_name() == "native-dist":
            rank = dist.get_rank()
            assert idist.get_rank() == rank

    assert idist.get_world_size() == ws
    assert idist.get_local_rank() == local_rank

    assert idist.model_name() in ("native-dist", "xla-dist")

    from ignite.distributed.utils import _model
    from ignite.distributed.comp_models import _NativeDistModel, _XlaDistModel

    _sanity_check()
    assert isinstance(_model, (_NativeDistModel, _XlaDistModel))


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


@pytest.mark.skipif(has_xla_support, reason="Skip if has PyTorch XLA package")
def test_xla_distrib_spawn_no_xla_support():
    with pytest.raises(RuntimeError, match=r"Torch xla package is not installed"):
        idist.spawn("xla-tpu", _test_native_distrib, args=("xla-tpu", 1, "xla"), num_procs_per_node=1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_spawn_one_proc():
    try:
        idist.spawn("xla-tpu", _test_native_distrib, args=("xla-tpu", 1, "xla"), num_procs_per_node=1)
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_spawn_n_procs():
    n = int(os.environ["NUM_TPU_WORKERS"])
    try:
        idist.spawn("xla-tpu", _test_native_distrib, args=("xla-tpu", n, "xla"), num_procs_per_node=n)
    except SystemExit:
        pass
