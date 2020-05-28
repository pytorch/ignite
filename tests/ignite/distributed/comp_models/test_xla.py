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
            _test_xla_spawn_fn, args=(1, "xla"), kwargs_dict={}, num_procs_per_node=1,
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
            _test_xla_spawn_fn, args=(n, "xla"), kwargs_dict={}, num_procs_per_node=n,
        )
    except SystemExit:
        pass


def _assert_model(model, true_conf):

    assert model.device() == true_conf["device"]
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_num_nodes() == true_conf["num_nodes"]
    assert model.get_ntasks_per_node() == true_conf["ntasks_per_node"]


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_create_from_backend():
    # without spawn
    model = _XlaDistModel.create_from_backend("xla-tpu")

    import torch_xla.core.xla_model as xm

    _assert_model(
        model,
        {
            "device": xm.xla_device(),
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": 1,
        },
    )

    model.finalize()


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_create_from_context():
    # without spawn
    model = _XlaDistModel.create_from_context()

    assert model.backend() == "xla-tpu"

    import torch_xla.core.xla_model as xm

    _assert_model(
        model,
        {
            "device": xm.xla_device(),
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": 1,
        },
    )


def _test__xla_dist_model_create_from_context_in_child_proc(index):
    model = _XlaDistModel.create_from_context()

    assert model.backend() == "xla-tpu"

    import torch_xla.core.xla_model as xm

    _assert_model(
        model,
        {
            "device": xm.xla_device(),
            "local_rank": index,
            "rank": xm.get_ordinal(),
            "world_size": xm.xrt_world_size(),
            "node_index": 0,
            "num_nodes": 1,
            "ntasks_per_node": xm.xrt_world_size(),
        },
    )


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_create_from_context_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test__xla_dist_model_create_from_context_in_child_proc, args=(), nprocs=n)
