import torch
import torch.distributed as dist

import ignite.distributed as idist
from tests.ignite.distributed.utils import _sanity_check, _test_sync


def test_no_distrib(capsys):

    assert idist.backend() is None
    if torch.cuda.is_available():
        assert idist.device().type == "cuda"
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
    assert idist.backend() is None, "{}".format(idist.backend())


def test_idist_all_reduce_no_dist():
    assert idist.all_reduce(10) == 10


def test_idist_all_gather_no_dist():
    assert idist.all_gather(10) == [10]
    assert (idist.all_gather(torch.tensor(10)) == torch.tensor(10)).all()
