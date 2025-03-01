import os

import pytest

import ignite.distributed as idist
from ignite.distributed.utils import has_xla_support
from tests.ignite.distributed.utils import (
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


@pytest.mark.skipif(has_xla_support, reason="Skip if has PyTorch XLA package")
def test_xla_distrib_spawn_no_xla_support():
    with pytest.raises(ValueError, match=r"Backend should be one of"):
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", 1, "xla"), nproc_per_node=1)


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
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", 1, "xla"), nproc_per_node=1)
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_xla_distrib_single_node_spawn_n_procs():
    n = int(os.environ["NUM_TPU_WORKERS"])
    try:
        idist.spawn("xla-tpu", _test_distrib_config, args=("xla-tpu", n, "xla"), nproc_per_node=n)
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_sync_as_xla():
    from ignite.distributed.comp_models.xla import _XlaDistModel

    _test_sync(_XlaDistModel)


def _test_sync_as_xla_in_child_proc(index):
    from ignite.distributed.comp_models.xla import _XlaDistModel

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
    from ignite.distributed.utils import _SerialModel, _set_model

    _set_model(_SerialModel())

    _test_distrib_config(local_rank=0, backend="xla-tpu", ws=1, true_device="xla", rank=0)


def _test_idist_methods_in_xla_context_in_child_proc(index):
    # We explicitly set _model as _SerialModel
    # then call idist.* methods and check that they give correct values
    from ignite.distributed.utils import _SerialModel, _set_model

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


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla():
    device = idist.device()
    _test_distrib_all_reduce(device)
    _test_distrib_all_reduce_group(device)


def _test_idist_all_reduce_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_all_reduce(device)
    _test_distrib_all_reduce_group(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_reduce_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_all_reduce_xla_in_child_proc, args=(), nprocs=n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_new_group_xla():
    device = idist.device()
    _test_distrib_group(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_gather_xla():
    device = idist.device()
    _test_distrib_all_gather(device)
    _test_distrib_all_gather_group(device)
    _test_idist_all_gather_tensors_with_shapes(device)
    _test_idist_all_gather_tensors_with_shapes_group(device)


def _test_idist_all_gather_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_all_gather(device)
    _test_distrib_all_gather_group(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_all_gather_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_all_gather_xla_in_child_proc, args=(), nprocs=n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_broadcast_xla():
    device = idist.device()
    _test_distrib_broadcast(device)


def _test_idist_broadcast_xla_in_child_proc(index):
    device = idist.device()
    _test_distrib_broadcast(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_broadcast_xla_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_idist_broadcast_xla_in_child_proc, args=(), nprocs=n)


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
