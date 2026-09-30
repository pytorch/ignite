import torch

import ignite.distributed as idist
from ignite.distributed.comp_models.base import _torch_version_gt_112
from tests.ignite.distributed.utils import (
    _sanity_check,
    _test_distrib__get_max_length,
    _test_distrib_all_gather,
    _test_distrib_all_reduce,
    _test_distrib_barrier,
    _test_distrib_broadcast,
    _test_distrib_group,
    _test_idist_all_gather_tensors_with_shapes,
    _test_sync,
)


def test_no_distrib(capsys):
    assert idist.backend() is None
    if torch.cuda.is_available():
        assert idist.device().type == "cuda"
    elif _torch_version_gt_112 and torch.backends.mps.is_available():
        assert idist.device().type == "mps"
    else:
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
    if torch.cuda.is_available():
        assert "ignite.distributed.utils INFO: device: cuda" in out[-1]
    elif _torch_version_gt_112 and torch.backends.mps.is_available():
        assert "ignite.distributed.utils INFO: device: mps" in out[-1]
    else:
        assert "ignite.distributed.utils INFO: device: cpu" in out[-1]
    assert "ignite.distributed.utils INFO: rank: 0" in out[-1]
    assert "ignite.distributed.utils INFO: local rank: 0" in out[-1]
    assert "ignite.distributed.utils INFO: world size: 1" in out[-1]


def test_sync_no_dist():
    from ignite.distributed.comp_models import _SerialModel

    _test_sync(_SerialModel)


def test_idist_methods_no_dist():
    assert idist.get_world_size() < 2
    assert idist.backend() is None, f"{idist.backend()}"


def test_idist__model_methods_no_dist():
    _test_distrib__get_max_length("cpu")
    if torch.cuda.device_count() > 1:
        _test_distrib__get_max_length("cuda")


def test_idist_collective_ops_no_dist():
    _test_distrib_all_reduce("cpu")
    _test_distrib_all_gather("cpu")
    _test_idist_all_gather_tensors_with_shapes("cpu")
    _test_distrib_barrier("cpu")
    _test_distrib_broadcast("cpu")
    _test_distrib_group("cpu")

    if torch.cuda.device_count() > 1:
        _test_distrib_all_reduce("cuda")
        _test_distrib_all_gather("cuda")
        _test_idist_all_gather_tensors_with_shapes("cuda")
        _test_distrib_barrier("cuda")
        _test_distrib_broadcast("cuda")
        _test_distrib_group("cuda")
