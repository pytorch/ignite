import os

import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import has_xla_support, sync
from ignite.engine import Engine, Events


def _sanity_check():
    from ignite.distributed.utils import _model

    assert _model.get_world_size() == _model.get_num_nodes() * _model.get_ntasks_per_node()
    assert _model.get_local_rank() < _model.get_ntasks_per_node()
    assert _model.get_rank() < _model.get_world_size()
    assert _model.get_node_rank() < _model.get_num_nodes()


def test_no_distrib(capsys):

    from ignite.distributed.utils import _model

    print("test_no_distrib : dist: ", dist.is_available(), dist.is_initialized())
    print("test_no_distrib : _model", type(_model))

    assert idist.backend() is None
    assert idist.device().type == "cpu"
    assert idist.get_rank() == 0
    assert idist.get_world_size() == 1
    assert idist.get_local_rank() == 0
    assert idist.model_name() == "serial"

    from ignite.distributed.utils import _model, _SerialModel

    _sanity_check()
    assert isinstance(_model, _SerialModel)

    idist.show_config()
    captured = capsys.readouterr()
    out = captured.err.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    assert "ignite.distributed.utils INFO: distributed configuration: serial" in out[-1]
    assert "ignite.distributed.utils INFO: backend: None" in out[-1]
    assert "ignite.distributed.utils INFO: device: cpu" in out[-1]
    assert "ignite.distributed.utils INFO: rank: 0" in out[-1]
    assert "ignite.distributed.utils INFO: local rank: 0" in out[-1]
    assert "ignite.distributed.utils INFO: world size: 1" in out[-1]


def _test_distrib_config(local_rank, backend, ws, true_device, rank=None):
    assert idist.backend() == backend, "{} vs {}".format(idist.backend(), backend)

    this_device = idist.device()
    assert isinstance(this_device, torch.device)
    if backend == "nccl":
        true_device = torch.device("{}:{}".format(true_device, local_rank))
        assert this_device == true_device, "{} vs {}".format(this_device, true_device)
    elif backend == "gloo":
        assert this_device == torch.device(true_device)
    elif backend == "xla-tpu":
        assert true_device in this_device.type

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
    _test_distrib_config(local_rank=0, backend="xla-tpu", ws=1, true_device="xla")
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


def _test_sync_as_xla_in_child_proc(index):
    from ignite.distributed.comp_models import _XlaDistModel

    _test_sync(_XlaDistModel)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_sync_as_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_sync_as_xla_in_child_proc, args=(), nprocs=n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_methods_in_xla_context():
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    _test_distrib_config(local_rank=0, backend="xla-tpu", ws=1, true_device="xla", rank=0)


def _test_idist_methods_in_xla_context_in_child_proc(index):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    import torch_xla.core.xla_model as xm

    _test_distrib_config(
        local_rank=index, backend="xla-tpu", ws=xm.xrt_world_size(), true_device="xla", rank=xm.get_ordinal()
    )


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_methods_in_xla_context_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_methods_in_xla_context_in_child_proc, args=(), nprocs=n)


def _test_idist_methods_in_native_context(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    ws = dist.get_world_size()
    rank = dist.get_rank()
    _test_distrib_config(local_rank, backend=backend, ws=ws, true_device=device, rank=rank)


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

    _test_distrib_config(local_rank=local_rank, backend=backend, ws=ws, true_device=device, rank=rank)

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


def test_idist_all_reduce_no_dist():
    assert idist.all_reduce(10) == 10


def test_idist_all_gather_no_dist():
    assert idist.all_gather(10) == 10


def _test_distrib_all_reduce(device):

    res = idist.all_reduce(10)
    assert res == 10 * idist.get_world_size()

    t = torch.tensor(10, device=device)
    res = idist.all_reduce(t)
    assert res.item() == 10 * idist.get_world_size()

    t = torch.tensor(idist.get_rank(), device=device)
    res = idist.all_reduce(t)
    assert res.item() == sum([i for i in range(idist.get_world_size())])

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_reduce("abc")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_reduce_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_all_reduce(device)


@pytest.mark.distributed
def test_idist_all_reduce_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_all_reduce(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla():
    device = idist.device()
    _test_distrib_all_reduce(device)


def _test_idist_all_reduce_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_all_reduce(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_all_reduce_xla_in_child_proc, args=(), nprocs=n)


def _test_distrib_all_gather(device):

    res = idist.all_gather(10)
    true_res = torch.tensor([10,] * idist.get_world_size(), device=device)
    assert (res == true_res).all()

    t = torch.tensor(idist.get_rank(), device=device)
    res = idist.all_gather(t)
    true_res = torch.tensor([i for i in range(idist.get_world_size())], device=device)
    assert (res == true_res).all()

    if idist.get_world_size() > 1:
        with pytest.raises(TypeError, match=r"Unhandled input type"):
            idist.all_gather("abc")

    t = torch.arange(100, device=device).reshape(4, 25) * (idist.get_rank() + 1)
    res = idist.all_gather(t)
    assert res.shape == (idist.get_world_size() * 4, 25)
    true_res = torch.zeros(idist.get_world_size() * 4, 25, device=device)
    for i in range(idist.get_world_size()):
        true_res[i * 4 : (i + 1) * 4, ...] = torch.arange(100, device=device).reshape(4, 25) * (i + 1)
    assert (res == true_res).all()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_gather_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_all_gather(device)


@pytest.mark.distributed
def test_idist_all_gather_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_all_gather(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_gather_xla():

    device = idist.device()
    _test_distrib_all_gather(device)


def _test_idist_all_gather_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_all_gather(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_gather_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_all_gather_xla_in_child_proc, args=(), nprocs=n)


def _test_distrib_barrier(device):

    t = torch.tensor([idist.get_rank()], device=device, dtype=torch.float)
    true_res = sum([i for i in range(idist.get_world_size())])

    if idist.get_rank() == 0:
        t += 10.0
    idist.barrier()

    tt = idist.all_reduce(t)
    assert tt.item() == true_res + 10.0


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_barrier_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_barrier(device)


@pytest.mark.distributed
def test_idist_barrier_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_barrier(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_barrier_xla():

    device = idist.device()
    _test_distrib_barrier(device)


def _test_idist_barrier_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_barrier(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_barrier_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_barrier_xla_in_child_proc, args=(), nprocs=n)


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


def _test_distrib_one_rank_only(device):
    def _test(barrier):
        # last rank
        rank = idist.get_world_size() - 1

        value = torch.tensor(0).to(device)

        @idist.one_rank_only(rank=rank, with_barrier=barrier)
        def initialize():
            value.data = torch.tensor(100).to(device)

        initialize()

        value_list = idist.all_gather(tensor=value)

        for r in range(idist.get_world_size()):
            if r == rank:
                assert value_list[r].item() == 100
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)


def _test_distrib_one_rank_only_with_engine(device):
    def _test(barrier):
        engine = Engine(lambda e, b: b)

        batch_sum = torch.tensor(0).to(device)

        @engine.on(Events.ITERATION_COMPLETED)
        @idist.one_rank_only(with_barrier=barrier)  # ie rank == 0
        def _(_):
            batch_sum.data += torch.tensor(engine.state.batch).to(device)

        engine.run([1, 2, 3], max_epochs=2)

        value_list = idist.all_gather(tensor=batch_sum)

        for r in range(idist.get_world_size()):
            if r == 0:
                assert value_list[r].item() == 12
            else:
                assert value_list[r].item() == 0

    _test(barrier=True)
    _test(barrier=False)


@pytest.mark.distributed
def test_idist_one_rank_only_gloo(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_one_rank_only_nccl(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_one_rank_only_xla():

    device = idist.device()
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


def _test_idist_one_rank_only_xla_nprocs(index):
    device = idist.device()
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_one_rank_only_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_one_rank_only_xla_nprocs, args=(), nprocs=n)
