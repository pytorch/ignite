import pytest
import torch

from ignite.distributed.comp_models.base import _SerialModel


def test_serial_model():
    _SerialModel.create_from_backend()
    model = _SerialModel.create_from_context()

    assert model.get_local_rank() == 0
    assert model.get_rank() == 0
    assert model.get_world_size() == 1
    assert model.get_nproc_per_node() == 1
    assert model.get_nnodes() == 1
    assert model.get_node_rank() == 0
    if torch.cuda.is_available():
        assert model.device().type == "cuda"
    else:
        assert model.device().type == "cpu"
    assert model.backend() is None
    model.finalize()

    with pytest.raises(NotImplementedError, match=r"Serial computation model does not implement spawn method"):
        model.spawn()

    model.all_reduce(1)
    model.all_gather(1)
    model._do_all_reduce(torch.tensor(1))
    model._do_all_gather(torch.tensor(1))
    model.barrier()
