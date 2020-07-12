import os

import pytest
import torch
import torch.distributed as dist

import ignite.distributed as idist
from ignite.distributed.utils import has_native_dist_support
from tests.ignite.distributed.utils import (
    _test_distrib_all_gather,
    _test_distrib_all_reduce,
    _test_distrib_barrier,
    _test_distrib_config,
    _test_distrib_one_rank_only,
    _test_distrib_one_rank_only_with_engine,
    _test_sync,
)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
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
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_launch_tool_nccl(local_rank, world_size):
    import os

    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.initialize("nccl")
    _test_distrib_config(local_rank, "nccl", world_size, "cuda", rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_native_distrib_single_node_spawn_gloo():

    from datetime import timedelta

    timeout = timedelta(seconds=20)

    world_size = 4

    idist.spawn(
        "gloo", _test_distrib_config, args=("gloo", world_size, "cpu"), nproc_per_node=world_size, timeout=timeout
    )


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_native_distrib_single_node_spawn_nccl():
    world_size = torch.cuda.device_count()

    idist.spawn("nccl", _test_distrib_config, args=("nccl", world_size, "cuda"), nproc_per_node=world_size)


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


def _test_idist_methods_in_native_context(backend, device, local_rank):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _set_model, _SerialModel

    _set_model(_SerialModel())

    ws = dist.get_world_size()
    rank = dist.get_rank()
    _test_distrib_config(local_rank, backend=backend, ws=ws, true_device=device, rank=rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_methods_in_native_gloo_context(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    _test_idist_methods_in_native_context("gloo", "cpu", local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
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
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_methods_in_native_gloo_context_set_local_rank(distributed_context_single_node_gloo):
    local_rank = distributed_context_single_node_gloo["local_rank"]
    _test_idist_methods_in_native_context_set_local_rank("gloo", "cpu", local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_in_native_nccl_context_set_local_rank(distributed_context_single_node_nccl):
    local_rank = distributed_context_single_node_nccl["local_rank"]
    _test_idist_methods_in_native_context_set_local_rank("nccl", "cuda", local_rank)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_reduce_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_all_reduce(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_all_reduce_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_all_reduce(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_all_gather_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_all_gather(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_all_gather_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_all_gather(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_barrier_nccl(distributed_context_single_node_nccl):

    device = "cuda:{}".format(distributed_context_single_node_nccl["local_rank"])
    _test_distrib_barrier(device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_barrier_gloo(distributed_context_single_node_gloo):

    device = "cpu"
    _test_distrib_barrier(device)


def _test_idist_methods_overhead(ok_factor):
    import time

    n = 100000
    m = 5

    t2 = 0.0
    t1 = 0.0
    for j in range(m):
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
    assert overhead_factor < ok_factor, "{} vs {} | {} vs {}".format(overhead_factor, ok_factor, t2, t1)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Do not want to run this test on Github or Travis, but CircleCI")
def test_idist_methods_overhead_gloo(distributed_context_single_node_gloo):
    _test_idist_methods_overhead(2.5)

    idist.sync()
    from ignite.distributed.utils import _model
    from ignite.distributed.comp_models.native import _NativeDistModel

    assert isinstance(_model, _NativeDistModel)

    _test_idist_methods_overhead(1.5)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_methods_overhead_nccl(distributed_context_single_node_nccl):
    _test_idist_methods_overhead(2.5)

    idist.sync()
    from ignite.distributed.utils import _model
    from ignite.distributed.comp_models.native import _NativeDistModel

    assert isinstance(_model, _NativeDistModel)

    _test_idist_methods_overhead(1.7)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_one_rank_only_gloo(distributed_context_single_node_gloo):
    device = "cpu"
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_one_rank_only_nccl(local_rank, distributed_context_single_node_nccl):
    device = "cuda:{}".format(local_rank)
    _test_distrib_one_rank_only(device=device)
    _test_distrib_one_rank_only_with_engine(device=device)
