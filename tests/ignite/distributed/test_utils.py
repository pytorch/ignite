import pytest

import ignite.distributed as idist


def test_no_distrib():
    assert idist.backend() is None
    assert idist.device() == "cpu"
    assert idist.get_rank() == 0
    assert idist.get_world_size() == 1
    assert idist.get_local_rank() == 0
    assert not idist.is_distributed()
    assert idist.is_initialized()
    assert idist.model_name() == "serial"

    from ignite.distributed.utils import _model, _SerialModel, _sanity_check

    _sanity_check()
    assert isinstance(_model, _SerialModel)


def _test_gloo_cpu_native_distrib(local_rank, rank, ws):
    assert idist.backend() == "gloo"
    assert idist.device() == "cpu"
    assert idist.get_rank() == rank
    assert idist.get_world_size() == ws
    assert idist.get_local_rank() == local_rank
    assert idist.is_distributed()
    assert idist.is_initialized()
    assert idist.model_name() == "native-dist"

    from ignite.distributed.utils import _model, _sanity_check
    from ignite.distributed.comp_models import _DistModel

    _sanity_check()
    assert isinstance(_model, _DistModel)


@pytest.mark.distributed
def test_gloo_cpu_native_distrib_single_node_launch_tool(local_rank, world_size):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.initialize("gloo", timeout=timeout)
    _test_gloo_cpu_native_distrib(local_rank, rank, world_size)
    idist.finalize()


@pytest.mark.distributed
def test_gloo_cpu_native_distrib_single_node_spawn(local_rank, world_size):

    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    rank = local_rank
    os.environ["RANK"] = "{}".format(rank)

    idist.spawn(
        _test_gloo_cpu_native_distrib,
        args=(local_rank, rank, world_size),
        backend="gloo",
        num_workers_per_machine=world_size,
        num_machines=1,
        machine_rank=rank,
        timeout=timeout,
    )
