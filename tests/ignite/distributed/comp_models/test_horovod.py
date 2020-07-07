import os

import pytest
import torch

from ignite.distributed.comp_models import has_hvd_support

if not has_hvd_support:
    pytest.skip("Skip if no Horovod package", allow_module_level=True)
else:
    from ignite.distributed.comp_models.horovod import _HorovodDistModel

import horovod.torch as hvd


def _gloo_hvd_execute(func, args, np=1):
    from horovod.run.runner import run

    return run(
        func, args=args, use_gloo=True, np=np
    )


@pytest.fixture()
def gloo_hvd_executor():
    yield _gloo_hvd_execute


@pytest.mark.distributed
def test__hvd_dist_model():
    with pytest.raises(ValueError, match=r"Backend should be one of"):
        _HorovodDistModel.create_from_backend("abc")


def _assert_model(model, true_conf):

    if "cuda" in true_conf["device"]:
        assert model.device() == torch.device("{}:{}".format(true_conf["device"], true_conf["local_rank"]))
    else:
        assert model.device() == torch.device(true_conf["device"])
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_nnodes() == true_conf["nnodes"]
    assert model.get_nproc_per_node() == true_conf["nproc_per_node"]


def _test__hvd_dist_model_create_from_backend_no_dist(backend, true_device):

    model = _HorovodDistModel.create_from_backend(backend=backend)

    assert hvd.rank() > -1

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": 0,
            "rank": 0,
            "world_size": 1,
            "node_index": 0,
            "nnodes": 1,
            "nproc_per_node": 1,
        },
    )

    model.finalize()


def _test__hvd_dist_model_create_from_backend_dist(backend, true_device):

    model = _HorovodDistModel.create_from_backend(backend=backend)

    assert hvd.rank() > -1

    with pytest.raises(RuntimeError, match=r"Can not re-initialize Horovod if it is already initialized"):
        _HorovodDistModel.create_from_backend(backend=backend)

    if "cuda" in true_device:
        true_device += ":{}".format(hvd.local_rank())

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": hvd.local_rank(),
            "rank": hvd.rank(),
            "world_size": hvd.size(),
            "node_index": 0,
            "nnodes": 1,
            "nproc_per_node": hvd.local_size(),
        },
    )

    model.finalize()


def _test__hvd_dist_model_create_from_context_no_dist(true_backend, true_device):

    with pytest.raises(ValueError, match=r"Horovod has not been initialized"):
        hvd.rank()

    assert _HorovodDistModel.create_from_context() is None

    hvd.init()

    true_conf = {
        "device": true_device,
        "local_rank": 0,
        "rank": 0,
        "world_size": 1,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": 1,
    }

    model = _HorovodDistModel.create_from_context()
    assert model.backend() == true_backend
    _assert_model(model, true_conf)

    hvd.shutdown()


def _test__native_dist_model_create_from_context_dist(true_backend, true_device):

    assert _HorovodDistModel.create_from_context() is None

    hvd.init()

    true_conf = {
        "device": true_device,
        "local_rank": hvd.local_rank(),
        "rank": hvd.rank(),
        "world_size": hvd.size(),
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": hvd.local_size(),
    }

    model = _HorovodDistModel.create_from_context()
    _assert_model(model, true_conf)

    hvd.shutdown()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Skip if has GPU")
def test__hvd_dist_model_create_no_dist(gloo_hvd_executor):
    gloo_hvd_executor(_test__hvd_dist_model_create_from_backend_no_dist, ("horovod", "cpu"), np=1)
    gloo_hvd_executor(_test__hvd_dist_model_create_from_context_no_dist, ("horovod", "cpu"), np=1)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_no_dist_cuda(gloo_hvd_executor):
    gloo_hvd_executor(_test__hvd_dist_model_create_from_backend_no_dist, ("horovod", "cuda:0"), np=1)
    gloo_hvd_executor(_test__hvd_dist_model_create_from_context_no_dist, ("horovod", "cuda:0"), np=1)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Skip if has GPU")
def test__native_dist_model_create_dist(gloo_hvd_executor):
    gloo_hvd_executor(
        _test__hvd_dist_model_create_from_backend_dist, ("horovod", "cpu"), np=4
    )
    gloo_hvd_executor(
        _test__native_dist_model_create_from_context_dist, ("horovod", "cpu"), np=4
    )


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_dist_cuda(gloo_hvd_executor):
    gloo_hvd_executor(
        _test__hvd_dist_model_create_from_backend_dist, ("horovod", "cuda"), np=torch.cuda.device_count()
    )
    gloo_hvd_executor(
        _test__native_dist_model_create_from_context_dist, ("horovod", "cuda"), np=torch.cuda.device_count()
    )


def _test_dist_spawn_fn(local_rank, backend, world_size, device):
    from ignite.distributed.utils import _model

    assert hvd.rank() > -1

    assert isinstance(_model, _HorovodDistModel), "{} vs _HorovodDistModel".format(type(_model))

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    assert _model.backend() == backend

    if "cuda" in device:
        assert _model.device() == torch.device("{}:{}".format(device, local_rank))
    else:
        assert _model.device() == torch.device(device)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Skip if has GPU")
def test__native_dist_model_spawn_gloo():
    num_workers_per_machine = 4
    _HorovodDistModel.spawn(
        _test_dist_spawn_fn,
        args=("horovod", num_workers_per_machine, "cpu"),
        kwargs_dict={},
        nproc_per_node=num_workers_per_machine,
        use_gloo=True
    )


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_spawn_nccl():
    num_workers_per_machine = torch.cuda.device_count()
    _HorovodDistModel.spawn(
        _test_dist_spawn_fn,
        args=("horovod", num_workers_per_machine, "cuda"),
        kwargs_dict={},
        nproc_per_node=num_workers_per_machine,
        use_gloo=True
    )
