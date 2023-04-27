import os

import pytest
import torch

from ignite.distributed.comp_models import has_xla_support

if not has_xla_support:
    pytest.skip("Skip if no XLA support", allow_module_level=True)
else:
    from ignite.distributed.comp_models.xla import _XlaDistModel


@pytest.mark.tpu
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_model():
    available_backends = _XlaDistModel.available_backends
    assert "xla-tpu" in available_backends

    with pytest.raises(ValueError, match=r"Backend should be one of"):
        _XlaDistModel.create_from_backend("abc")


def _test_xla_spawn_fn(local_rank, world_size, device):
    from ignite.distributed.utils import _model

    assert isinstance(_model, _XlaDistModel), f"{type(_model)} vs _XlaDistModel"

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    d = _model.device()
    assert isinstance(d, torch.device) and d.type == device

    assert _model.get_rank() == local_rank
    assert _model.get_nproc_per_node() == world_size
    assert _model.get_node_rank() == 0
    assert _model.get_nnodes() == 1


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_spawn_one_proc():
    try:
        _XlaDistModel.spawn(_test_xla_spawn_fn, args=(1, "xla"), kwargs_dict={}, nproc_per_node=1)
    except SystemExit:
        pass


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_spawn_n_procs():
    n = int(os.environ["NUM_TPU_WORKERS"])
    try:
        _XlaDistModel.spawn(_test_xla_spawn_fn, args=(n, "xla"), kwargs_dict={}, nproc_per_node=n)
    except SystemExit:
        pass


def _assert_model(model, true_conf):
    assert model.device() == true_conf["device"]
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_nnodes() == true_conf["nnodes"]
    assert model.get_nproc_per_node() == true_conf["nproc_per_node"]


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
            "nnodes": 1,
            "nproc_per_node": 1,
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
            "nnodes": 1,
            "nproc_per_node": 1,
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
            "nnodes": 1,
            "nproc_per_node": xm.xrt_world_size(),
        },
    )


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_create_from_context_in_child_proc(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test__xla_dist_model_create_from_context_in_child_proc, args=(), nprocs=n)


def main_fold(fold):
    import time

    import torch.nn as nn
    import torch.optim as optim
    import torch_xla.core.xla_model as xm

    from ignite.engine import Engine

    device = xm.xla_device(fold)

    comp_model = _XlaDistModel.create_from_context()
    assert comp_model.device() == device

    model = nn.Linear(100, 10)

    model.to(device)  # Move model before creating optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def training_step(engine, _):
        data = torch.rand(4, 100, device=device)
        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        return loss.item()

    trainer = Engine(training_step)

    # THIS CAN BE A CAUSE OF CRASH if DEVICE is OTHER THAN device
    tensor = torch.tensor([fold + 1.0], dtype=torch.float).to(comp_model.device())
    xm.all_reduce("max", [tensor])

    time.sleep(0.01 * fold)

    trainer.run([0] * 100, max_epochs=2)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test__xla_dist_model_run_parallel_n_threads_without_sync():
    # tests issue : https://github.com/pytorch/ignite/issues/1096
    import torch_xla.core.xla_model as xm
    from joblib import delayed, Parallel

    devices = xm.get_xla_supported_devices()
    folds = 1
    d = 0
    if len(devices) > 5:
        folds = 5
        d = 1
    Parallel(n_jobs=folds, backend="threading")(delayed(main_fold)(i + d) for i in range(folds))
