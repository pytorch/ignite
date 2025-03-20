import os

import pytest
import torch

import ignite.distributed as idist
from ignite.distributed.utils import has_hvd_support
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


@pytest.mark.skipif(has_hvd_support, reason="Skip if has Horovod package")
def test_hvd_distrib_spawn_no_hvd_support():
    with pytest.raises(ValueError, match=r"Backend should be one of"):
        idist.spawn("horovod", _test_distrib_config, args=("horovod", 1, "cpu"), nproc_per_node=1)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
def test_hvd_distrib_single_node_single_device():
    import horovod.torch as hvd

    idist.initialize("horovod")

    device = "cpu" if torch.cuda.device_count() < 1 else "cuda"
    local_rank = hvd.local_rank()
    world_size = hvd.size()
    rank = hvd.rank()
    _test_distrib_config(local_rank, "horovod", world_size, device, rank)
    idist.finalize()


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Skip if has GPU")
def test_hvd_distrib_single_node_spawn():
    world_size = 4

    idist.spawn("horovod", _test_distrib_config, args=("horovod", world_size, "cpu"), nproc_per_node=world_size)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_hvd_distrib_multi_node_spawn_raise_error():
    world_size = 4

    with pytest.raises(RuntimeError, match=r"For multi-node configuration, please set 'hosts' argument instead"):
        idist.spawn(
            "horovod", _test_distrib_config, args=("horovod", world_size, "cpu"), nproc_per_node=world_size, nnodes=2
        )


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_hvd_distrib_single_node_spawn_cuda():
    world_size = torch.cuda.device_count()

    idist.spawn("horovod", _test_distrib_config, args=("horovod", world_size, "cuda"), nproc_per_node=world_size)


def _test_sync_as_hvd():
    import horovod.torch as hvd

    from ignite.distributed.comp_models.horovod import _HorovodDistModel

    hvd.init()
    lrank = hvd.local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(lrank)

    _test_sync(_HorovodDistModel)

    hvd.shutdown()


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif(os.getenv("HOROVOD_RANK", -1) == -1, reason="Skip as controller is not Gloo")
def test_sync_as_hvd():
    _test_sync_as_hvd()


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_sync_as_hvd_inside_gloo_executor(gloo_hvd_executor):
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_sync_as_hvd, (), np=np)


def _test_idist_methods_in_hvd_context(backend, device):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    import horovod.torch as hvd

    from ignite.distributed.utils import _SerialModel, _set_model

    hvd.init()

    _set_model(_SerialModel())

    ws = hvd.size()
    rank = hvd.rank()
    local_rank = hvd.local_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    _test_distrib_config(local_rank, backend=backend, ws=ws, true_device=device, rank=rank)

    hvd.shutdown()


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_methods_in_hvd_context(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_idist_methods_in_hvd_context, ("horovod", device), np=np)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_all_reduce_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib_all_reduce, (device,), np=np, do_init=True)
    gloo_hvd_executor(_test_distrib_all_reduce_group, (device,), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist__model_methods_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib__get_max_length, (device,), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_all_gather_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib_all_gather, (device,), np=np, do_init=True)
    gloo_hvd_executor(_test_distrib_all_gather_group, (device,), np=np, do_init=True)
    gloo_hvd_executor(_test_idist_all_gather_tensors_with_shapes, (device,), np=np, do_init=True)
    gloo_hvd_executor(_test_idist_all_gather_tensors_with_shapes_group, (device,), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_broadcast_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib_broadcast, (device,), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_barrier_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib_barrier, (device,), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_new_group_hvd(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    gloo_hvd_executor(_test_distrib_group, (device,), np=np, do_init=True)


def _test_idist_methods_overhead(ok_factor, sync_model):
    import time

    import horovod.torch as hvd

    if sync_model:
        idist.sync()
        from ignite.distributed.comp_models.horovod import _HorovodDistModel
        from ignite.distributed.utils import _model

        assert isinstance(_model, _HorovodDistModel)

    n = 100000
    m = 5

    t2 = 0.0
    t1 = 0.0
    for _ in range(m):
        start = time.time()
        for _ in range(n):
            _ = hvd.size()
            _ = hvd.rank()
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
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_methods_overhead_hvd(gloo_hvd_executor):
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    ok_factor = 6.0
    sync_model = False
    gloo_hvd_executor(_test_idist_methods_overhead, (ok_factor, sync_model), np=np, do_init=True)

    ok_factor = 3.5
    sync_model = True
    gloo_hvd_executor(_test_idist_methods_overhead, (ok_factor, sync_model), np=np, do_init=True)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_one_rank_only(gloo_hvd_executor):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_one_rank_only, (device,), np=np, do_init=True)
    gloo_hvd_executor(_test_distrib_one_rank_only_with_engine, (device,), np=np, do_init=True)
