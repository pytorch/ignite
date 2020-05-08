import os

import pytest

import torch

from ignite.distributed.comp_models import _XlaDistModel, has_xla_support


@pytest.mark.tpu
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_model():
    available_backends = _XlaDistModel.available_backends
    assert "xla-tpu" in available_backends


def _test_xla_spawn_fn(local_rank, world_size, device):
    from ignite.distributed.utils import _model

    assert isinstance(_model, _XlaDistModel), "{} vs _XlaDistModel".format(type(_model))

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    d = _model.device()
    assert isinstance(d, torch.device) and d.type == device

    assert _model.get_rank() == local_rank
    assert _model.get_ntasks_per_node() == world_size
    assert _model.get_node_rank() == 0
    assert _model.get_num_nodes() == 1


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_spawn_one_proc():
    try:
        _XlaDistModel.spawn(
            _test_xla_spawn_fn, args=(1, "xla"), num_procs_per_node=1,
        )
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_spawn_n_procs():
    n = int(os.environ["NUM_TPU_WORKERS"])
    try:
        _XlaDistModel.spawn(
            _test_xla_spawn_fn, args=(n, "xla"), num_procs_per_node=n,
        )
    except SystemExit:
        pass
