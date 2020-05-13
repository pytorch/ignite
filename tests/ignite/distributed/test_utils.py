import os

import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import has_xla_support, sync


def _sanity_check():
    from ignite.distributed.utils import _model

    assert _model.get_world_size() == _model.get_num_nodes() * _model.get_ntasks_per_node()
    assert _model.get_local_rank() < _model.get_ntasks_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node_rank() < _model.get_num_nodes()


def test_no_distrib():

    from ignite.distributed.utils import _model

    print("test_no_distrib : dist: ", dist.is_available(), dist.is_initialized())
    print("test_no_distrib : _model", type(_model))

    assert idist.backend() is None
    assert idist.device() == "cpu"
    assert idist.get_rank() == 0
    assert idist.get_world_size() == 1
    assert idist.get_local_rank() == 0
    assert idist.model_name() == "serial"

    from ignite.distributed.utils import _model, _SerialModel

    _sanity_check()
    assert isinstance(_model, _SerialModel)


def _test_distrib_config(local_rank, backend, ws, device, rank=None):
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
    _test_distrib_config(local_rank, "gloo", world_size, "cpu", rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_launch_tool_nccl(local_rank, world_size):
    import os

    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.initialize("nccl")
    _test_distrib_config(local_rank, "nccl", world_size, "cuda", rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_native_distrib_single_node_spawn_gloo():

    from datetime import timedelta

    timeout = timedelta(seconds=20)

    world_size = 4

    idist.spawn(
        "gloo", _test_distrib_config, args=("gloo", world_size, "cpu"), num_procs_per_node=world_size, timeout=timeout
    )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_spawn_nccl():
    world_size = torch.cuda.device_count()

    idist.spawn("nccl", _test_distrib_config, args=("nccl", world_size, "cuda"), num_procs_per_node=world_size)


@pytest.mark.skipif(has_xla_support, reason="Skip if has PyTorch XLA package")
def test_xla_distrib_spawn_no_xla_support():
    with pytest.raises(RuntimeError, match=r"Torch xla package is not installed"):
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", 1, "xla"), num_procs_per_node=1)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_no_spawn():
    idist.initialize("xla-tpu")
    _test_distrib_config(local_rank=0, backend="xla-tpu", ws=1, device="xla")
    idist.finalize()


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_spawn_one_proc():
    try:
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", 1, "xla"), num_procs_per_node=1)
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_spawn_n_procs():
    n = int(os.environ["NUM_TPU_WORKERS"])
    try:
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", n, "xla"), num_procs_per_node=n)
    except SystemExit:
        pass


def _test_sync(cls):
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    sync()

    from ignite.distributed.utils import _model

    assert isinstance(_model, cls), "{} vs {}".format(type(_model), cls)


def test_sync_no_dist():
    from ignite.distributed.comp_models import _SerialModel

    _test_sync(_SerialModel)


def test_idist_methods_no_dist():
    assert idist.get_world_size() < 2
    assert idist.backend() is None, "{}".format(idist.backend())


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_sync_as_xla():
    from ignite.distributed.comp_models import _XlaDistModel

    _test_sync(_XlaDistModel)


@pytest.mark.distributed
def test_sync_as_native_gloo(distributed_context_single_node_gloo):
    from ignite.distributed.comp_models import _NativeDistModel

    _test_sync(_NativeDistModel)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_sync_as_native_nccl(distributed_context_single_node_nccl):
    from ignite.distributed.comp_models import _NativeDistModel

    _test_sync(_NativeDistModel)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_sync_as_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])

    def _test_fn(index):
        from ignite.distributed.comp_models import _XlaDistModel

        _test_sync(_XlaDistModel)

    xmp_executor(_test_fn, args=(), nprocs=n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_methods_in_xla_context():
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    _test_distrib_config(local_rank=0, backend="xla-tpu", ws=1, device="xla", rank=0)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_methods_in_xla_context_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])

    def _test_fn(index):
        # We explicitly set _model as _SerialModel
        # then call idist.* methods and check that they give correct values
        from ignite.distributed.utils import _set_model, _SerialModel

        _set_model(_SerialModel())

        import torch_xla.core.xla_model as xm

        _test_distrib_config(
            local_rank=index, backend="xla-tpu", ws=xm.xrt_world_size(), device="xla", rank=xm.get_ordinal()
        )

    xmp_executor(_test_fn, args=(), nprocs=n)


def _test_idist_methods_in_native_context(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    ws = dist.get_world_size()
    rank = dist.get_rank()
    _test_distrib_config(local_rank=local_rank, backend=backend, ws=ws, device=device, rank=rank)


@pytest.mark.distributed
def test_idist_methods_in_native_gloo_context(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    _test_idist_methods_in_native_context("gloo", "cpu", local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_in_native_nccl_context(distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    _test_idist_methods_in_native_context("nccl", "cuda", local_rank)


def _test_idist_methods_in_native_context_set_local_rank(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    lrank = int(os.environ["LOCAL_RANK"])
    del os.environ["LOCAL_RANK"]

    ws = dist.get_world_size()
    rank = dist.get_rank()

    idist.set_local_rank(local_rank)

    _test_distrib_config(local_rank=local_rank, backend=backend, ws=ws, device=device, rank=rank)

    os.environ["LOCAL_RANK"] = str(lrank)


@pytest.mark.distributed
def test_idist_methods_in_native_gloo_context_set_local_rank(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    _test_idist_methods_in_native_context_set_local_rank("gloo", "cpu", local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_in_native_nccl_context_set_local_rank(distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    _test_idist_methods_in_native_context_set_local_rank("nccl", "cuda", local_rank)


def test_idist_all_reduce():
    assert idist.all_reduce(10) == 10


def _test_distrib__sync_all_reduce(device):

    res = idist.all_reduce(10)
    assert res == 10 * idist.get_world_size()

    t = torch.tensor(10, device=device)
    res = idist.all_reduce(t)
    assert res.item() == 10 * idist.get_world_size()

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_reduce("abc")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_reduce_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib__sync_all_reduce(device)


@pytest.mark.distributed
def test_idist_all_reduce_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib__sync_all_reduce(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla():
    device = idist.device()
    _test_distrib__sync_all_reduce(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])

    def _test_fn(index):
        device = idist.device()
        _test_distrib__sync_all_reduce(device)

    xmp_executor(_test_fn, args=(), nprocs=n)


@pytest.mark.distributed
def test_idist_methods_overhead_gloo(distributed_context_single_node_gloo):
    import time

    n = 100000
    start = time.time()
    for _ in range(n):
        _ = idist.get_world_size()
        _ = idist.get_rank()
    elapsed = time.time() - start
    t1 = elapsed / n

    start = time.time()
    for _ in range(n):
        _ = dist.get_world_size()
        _ = idist.get_rank()
    elapsed = time.time() - start
    t2 = elapsed / n

    assert t2 * 6 > t1, "{} * 6 vs {}".format(t2, t1)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_overhead_nccl(distributed_context_single_node_nccl):
    import time

    n = 100000
    start = time.time()
    for _ in range(n):
        _ = idist.get_world_size()
        _ = idist.get_rank()
    elapsed = time.time() - start
    t1 = elapsed / n

    start = time.time()
    for _ in range(n):
        _ = dist.get_world_size()
        _ = idist.get_rank()
    elapsed = time.time() - start
    t2 = elapsed / n

    assert t2 * 3 > t1, "{} * 3 vs {}".format(t2, t1)
