import os

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version

import ignite.distributed as idist
from ignite.distributed.utils import has_native_dist_support
from tests.ignite.distributed.utils import (
    _test_distrib__get_max_length,
    _test_distrib_all_gather,
    _test_distrib_all_gather_group,
    _test_distrib_all_reduce,
    _test_distrib_all_reduce_group,
    _test_distrib_barrier,
    _test_distrib_broadcast,
    _test_distrib_config,
    _test_distrib_group,
    _test_distrib_one_rank_only,
    _test_distrib_one_rank_only_with_engine,
    _test_idist_all_gather_tensors_with_shapes,
    _test_idist_all_gather_tensors_with_shapes_group,
    _test_sync,
)


def _test_native_distrib_single_node_launch_tool(backend, device, local_rank, world_size, init_method=None, **kwargs):
    import os

    rank = local_rank
    os.environ["RANK"] = f"{rank}"

    idist.initialize(backend, init_method=init_method, **kwargs)
    _test_distrib_config(local_rank, backend, world_size, device, rank, true_init_method=init_method)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test_native_distrib_single_node_launch_tool_gloo(init_method, get_fixed_dirname, local_rank, world_size):
    from datetime import timedelta

    timeout = timedelta(seconds=20)

    if init_method == "FILE":
        init_method = f"file://{get_fixed_dirname('native_distrib_single_node_launch_tool_gloo')}/shared"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _test_native_distrib_single_node_launch_tool(
        "gloo", device, local_rank, world_size, timeout=timeout, init_method=init_method
    )


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test_native_distrib_single_node_launch_tool_nccl(init_method, get_fixed_dirname, local_rank, world_size):
    if init_method == "FILE":
        init_method = f"file://{get_fixed_dirname('native_distrib_single_node_launch_tool_nccl')}/shared"

    device = torch.device(f"cuda:{local_rank}")
    _test_native_distrib_single_node_launch_tool("nccl", device, local_rank, world_size, init_method=init_method)


def _test_native_distrib_single_node_spawn(init_method, backend, device, **kwargs):
    world_size = 4 if torch.device(device).type == "cpu" else torch.cuda.device_count()
    idist.spawn(
        backend,
        _test_distrib_config,
        args=(backend, world_size, device),
        nproc_per_node=world_size,
        init_method=init_method,
        **kwargs,
    )


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test_native_distrib_single_node_spawn_gloo(init_method, dirname):
    from datetime import timedelta

    timeout = timedelta(seconds=20)

    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _test_native_distrib_single_node_spawn(init_method, "gloo", device, timeout=timeout)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test_native_distrib_single_node_spawn_nccl(init_method, dirname):
    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"

    device = torch.device("cuda")
    _test_native_distrib_single_node_spawn(init_method, "nccl", device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_sync_as_native_gloo(distributed_context_single_node_gloo):
    from ignite.distributed.comp_models.native import _NativeDistModel

    _test_sync(_NativeDistModel)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_sync_as_native_nccl(distributed_context_single_node_nccl):
    from ignite.distributed.comp_models.native import _NativeDistModel

    _test_sync(_NativeDistModel)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_new_group_native_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_new_group_native_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_group(device)


def _test_idist_methods_in_native_context(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _SerialModel, _set_model

    _set_model(_SerialModel())

    ws = dist.get_world_size()
    rank = dist.get_rank()
    _test_distrib_config(local_rank, backend=backend, ws=ws, true_device=device, rank=rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_methods_in_native_gloo_context(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _test_idist_methods_in_native_context("gloo", device, local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_in_native_nccl_context(distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    device = torch.device(f"cuda:{local_rank}")
    _test_idist_methods_in_native_context("nccl", device, local_rank)


def _test_idist_methods_in_native_context_set_local_rank(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _SerialModel, _set_model

    _set_model(_SerialModel())

    lrank = int(os.environ["LOCAL_RANK"])
    del os.environ["LOCAL_RANK"]

    ws = dist.get_world_size()
    rank = dist.get_rank()

    idist.set_local_rank(local_rank)

    _test_distrib_config(local_rank=local_rank, backend=backend, ws=ws, true_device=device, rank=rank)

    os.environ["LOCAL_RANK"] = str(lrank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_methods_in_native_gloo_context_set_local_rank(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    device = idist.device()
    _test_idist_methods_in_native_context_set_local_rank("gloo", device, local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_in_native_nccl_context_set_local_rank(distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    device = idist.device()
    _test_idist_methods_in_native_context_set_local_rank("nccl", device, local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist__model_methods_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib__get_max_length(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist__model_methods_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib__get_max_length(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_reduce_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_all_reduce(device)
    if idist.get_world_size() > 1:
        _test_distrib_all_reduce_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_all_reduce_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_all_reduce(device)
    if idist.get_world_size() > 1:
        _test_distrib_all_reduce_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="dist.all_gather_object is not implemented")
def test_idist_all_gather_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_all_gather(device)
    if idist.get_world_size() > 1:
        _test_distrib_all_gather_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(Version(torch.__version__) < Version("1.7.0"), reason="dist.all_gather_object is not implemented")
def test_idist_all_gather_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_all_gather(device)
    if idist.get_world_size() > 1:
        _test_distrib_all_gather_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_gather_tensors_with_shapes_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_idist_all_gather_tensors_with_shapes(device)
    if idist.get_world_size() > 1:
        _test_idist_all_gather_tensors_with_shapes_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_all_gather_tensors_with_shapes_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_idist_all_gather_tensors_with_shapes(device)
    if idist.get_world_size() > 1:
        _test_idist_all_gather_tensors_with_shapes_group(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_broadcast_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_broadcast(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_broadcast_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_broadcast(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_barrier_nccl(distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_barrier(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_barrier_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_barrier(device)


def _test_idist_methods_overhead(ok_factor):
    import time

    n = 100000
    m = 5

    t2 = 0.0
    t1 = 0.0
    for _ in range(m):
        start = time.time()
        for _ in range(n):
            _ = dist.get_world_size()
            _ = dist.get_rank()
        elapsed = time.time() - start
        t2 += elapsed / n / m

        start = time.time()
        for _ in range(n):
            _ = idist.get_world_size()
            _ = idist.get_rank()
        elapsed = time.time() - start
        t1 += elapsed / n / m

    overhead_factor = t1 / t2
    assert overhead_factor < ok_factor, f"{overhead_factor} vs {ok_factor} | {t2} vs {t1}"


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Do not want to run this test on Github or Travis, but CircleCI"
)
def test_idist_methods_overhead_gloo(distributed_context_single_node_gloo):
    _test_idist_methods_overhead(2.5)

    idist.sync()
    from ignite.distributed.comp_models.native import _NativeDistModel
    from ignite.distributed.utils import _model

    assert isinstance(_model, _NativeDistModel)

    _test_idist_methods_overhead(1.7)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_overhead_nccl(distributed_context_single_node_nccl):
    _test_idist_methods_overhead(2.5)

    idist.sync()
    from ignite.distributed.comp_models.native import _NativeDistModel
    from ignite.distributed.utils import _model

    assert isinstance(_model, _NativeDistModel)

    _test_idist_methods_overhead(1.7)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_one_rank_only_gloo(distributed_context_single_node_gloo):
    device = idist.device()
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_one_rank_only_nccl(local_rank, distributed_context_single_node_nccl):
    device = idist.device()
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.distributed
@pytest.mark.parametrize("rank", range(int(os.environ.get("WORLD_SIZE", 1))))
@pytest.mark.parametrize("local", [True, False])
def test_one_rank_first(distributed, get_rank_zero_dirname, rank, local):
    def get_ds(file_path):
        rank = idist.get_local_rank() if local else idist.get_rank()
        if not file_path.exists():
            with open(file_path, "w") as f:
                f.write("readed")
            return f"{rank} not readed"
        else:
            return f"{rank} readed"

    folder = get_rank_zero_dirname()
    file_path = folder / "res.txt"

    with idist.one_rank_first(rank, local=local):
        x = get_ds(file_path)

    output = idist.all_gather(x)

    if local:
        expected = [
            f"{x} not readed" if x == rank else f"{x} readed" for x in range(idist.get_nproc_per_node())
        ] * idist.get_nnodes()
    else:
        expected = [f"{x} not readed" if x == rank else f"{x} readed" for x in range(idist.get_world_size())]

    print("expected:", expected, idist.get_nnodes())
    assert set(expected) == set(output)


@pytest.mark.distributed
def test_one_rank_first_asserts():
    rank = 100
    with pytest.raises(
        ValueError, match=f"rank should be between 0 and {idist.get_world_size() - 1}, but given {rank}"
    ):
        with idist.one_rank_first(rank):
            pass
