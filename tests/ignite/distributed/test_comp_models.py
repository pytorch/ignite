import pytest

import torch.distributed as dist

from ignite.distributed.comp_models import _SerialModel, _XlaDistModel, _DistModel


@pytest.mark.skipif(not dist.is_available(), reason="no torch dist available")
def test__dist_model():
    available_backends = _DistModel.available_backends

    if dist.is_nccl_available():
        assert "nccl" in available_backends

    if dist.is_gloo_available():
        assert "gloo" in available_backends

    if dist.is_mpi_available():
        assert "mpi" in available_backends


def test__dist_model_init():

    model = _DistModel(backend="gloo")

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == "gloo"
    assert model.device() == "cpu"
    assert model.get_local_rank() == 0
    assert model.get_world_size() == 1
    assert model.get_rank() == 0

    dist.destroy_process_group()
