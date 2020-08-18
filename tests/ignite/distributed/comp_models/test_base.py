import pytest
import torch

from ignite.distributed.comp_models.base import ComputationModel, _SerialModel


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
    model.broadcast(1)
    model._do_all_reduce(torch.tensor(1))
    model._do_all_gather(torch.tensor(1))
    model._do_broadcast(torch.tensor(1), src=0)
    model.barrier()


def test__encode_str__decode_str():
    device = torch.device("cpu")
    s = "test-abcedfg"

    encoded_s = ComputationModel._encode_str(s, device)
    assert isinstance(encoded_s, torch.Tensor) and encoded_s.shape == (1, 1025)

    decoded_s = ComputationModel._decode_str(encoded_s)
    assert isinstance(decoded_s, list) and len(decoded_s) == 1
    assert decoded_s[0] == s
