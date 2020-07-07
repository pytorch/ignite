import os

import pytest
import torch

import ignite.distributed as idist
from ignite.distributed.utils import has_hvd_support
from tests.ignite.distributed.utils import (
    _test_distrib_all_gather,
    _test_distrib_all_reduce,
    _test_distrib_barrier,
    _test_distrib_config,
    _test_distrib_one_rank_only,
    _test_distrib_one_rank_only_with_engine,
    _test_sync,
)


@pytest.mark.skipif(has_hvd_support, reason="Skip if has Horovod package")
def test_hvd_distrib_spawn_no_hvd_support():
    with pytest.raises(ValueError, match=r"Backend should be one of"):
        idist.spawn("horovod", _test_distrib_config, args=("horovod", 1, "cpu"), nproc_per_node=1)


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
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_hvd_distrib_single_node_spawn_cuda():
    world_size = torch.cuda.device_count()

    idist.spawn("horovod", _test_distrib_config, args=("horovod", world_size, "cuda"), nproc_per_node=world_size)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
def test_sync_as_hvd():
    from ignite.distributed.comp_models.horovod import _HorovodDistModel

    import horovod.torch as hvd

    hvd.init()

    _test_sync(_HorovodDistModel)

    hvd.shutdown()

# @pytest.mark.distributed
# @pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
# @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
# def test_idist_methods_in_hvd_context():
#
#     _test_idist_methods_in_native_context_set_local_rank("nccl", "cuda", local_rank)
