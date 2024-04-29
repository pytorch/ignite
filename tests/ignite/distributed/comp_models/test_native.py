import os

import pytest
import torch
import torch.distributed as dist

from ignite.distributed.comp_models import has_native_dist_support

if not has_native_dist_support:
    pytest.skip("Skip if no native dist support", allow_module_level=True)
else:
    from ignite.distributed.comp_models.native import _expand_hostlist, _NativeDistModel, _setup_ddp_vars_from_slurm_env


# tests from https://github.com/LLNL/py-hostlist/blob/master/hostlist/unittest_hostlist.py
@pytest.mark.parametrize(
    "hostlist, expected",
    [
        ("localhost", "localhost"),
        ("compute!:b24_[1-2].r", "compute!:b24_1.r,compute!:b24_2.r"),
        ("quartz[4-8]", "quartz4,quartz5,quartz6,quartz7,quartz8"),
        ("c1001a-[11,17]", "c1001a-11,c1001a-17"),
        ("c1001a-s[11,17]", "c1001a-s11,c1001a-s17"),
        ("c1009a-s17,c1010a-s11", "c1009a-s17,c1010a-s11"),
        (
            "gpu-compute-on-demand-dy-g4dnxlarge-[1-4]",
            "gpu-compute-on-demand-dy-g4dnxlarge-1,"
            "gpu-compute-on-demand-dy-g4dnxlarge-2,"
            "gpu-compute-on-demand-dy-g4dnxlarge-3,"
            "gpu-compute-on-demand-dy-g4dnxlarge-4",
        ),
        (
            "node[18-19,1-16,21-22]",
            "node1,node2,node3,node4,node5,"
            "node6,node7,node8,node9,node10,"
            "node11,node12,node13,node14,node15,"
            "node16,node18,node19,node21,node22",
        ),
        (
            "node[4-8,12,16-20,22,24-26]",
            "node4,node5,node6,node7,node8,"
            "node12,node16,node17,node18,"
            "node19,node20,node22,node24,"
            "node25,node26",
        ),
        ("machine2-[02-4]vm1", "machine2-02vm1,machine2-03vm1,machine2-04vm1"),
        (
            "machine2-[02-3]vm1, machine4-[0003-5].vml2",
            "machine2-02vm1,machine2-03vm1,machine4-0003.vml2,machine4-0004.vml2,machine4-0005.vml2",
        ),
        ("machine2-[009-11]vm1", "machine2-009vm1,machine2-010vm1,machine2-011vm1"),
        ("node[1,2,3]", "node1,node2,node3"),
        (
            "compute-b24-[1-3,5-9], compute-b25-[1,4,8],compute-b25-[2-9,13]",
            "compute-b24-1,compute-b24-2,compute-b24-3,compute-b24-5,compute-b24-6,"
            "compute-b24-7,compute-b24-8,compute-b24-9,compute-b25-1,compute-b25-4,"
            "compute-b25-8,compute-b25-2,compute-b25-3,compute-b25-4,compute-b25-5,"
            "compute-b25-6,compute-b25-7,compute-b25-8,compute-b25-9,compute-b25-13",
        ),
    ],
)
def test_expand_hostlist(hostlist, expected):
    assert _expand_hostlist(hostlist) == expected.split(",")


def test_expand_hostlist_invalid():
    with pytest.raises(ValueError, match=r"hostlist invalid"):
        _expand_hostlist("invalid[]")


@pytest.mark.distributed
def test__native_dist_model():
    available_backends = _NativeDistModel.available_backends

    if dist.is_nccl_available():
        assert "nccl" in available_backends
    else:
        assert "nccl" not in available_backends

    if dist.is_gloo_available():
        assert "gloo" in available_backends
    else:
        assert "gloo" not in available_backends

    if dist.is_mpi_available():
        assert "mpi" in available_backends
    else:
        assert "mpi" not in available_backends

    with pytest.raises(ValueError, match=r"Backend should be one of"):
        _NativeDistModel.create_from_backend("abc")


@pytest.mark.distributed
@pytest.mark.skipif(not dist.is_nccl_available(), reason="Skip if nccl not available")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_nccl_but_no_gpu(mock_gpu_is_not_available):
    with pytest.raises(RuntimeError, match=r"Nccl backend is required but no cuda capable devices"):
        _NativeDistModel(backend="nccl")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_dist_model_create_from_backend_bad_config():
    import os
    from datetime import timedelta

    os.environ["RANK"] = "1"

    with pytest.raises(RuntimeError, match=r"PyTorch distributed configuration should define env variables"):
        _NativeDistModel.create_from_backend(backend="gloo", timeout=timedelta(seconds=10))

    del os.environ["RANK"]


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test__native_dist_model_create_from_backend_bad_slurm_config():
    import os
    from datetime import timedelta

    os.environ["SLURM_JOB_ID"] = "1"

    with pytest.raises(RuntimeError, match=r"SLURM distributed configuration is missing"):
        _NativeDistModel.create_from_backend(backend="gloo", timeout=timedelta(seconds=10))

    with pytest.raises(ValueError, match=r"Arguments rank and world_size should not be specified with SLURM"):
        _NativeDistModel.create_from_backend(
            backend="gloo", timeout=timedelta(seconds=10), rank=1, init_method="", world_size=1
        )

    os.environ["SLURM_PROCID"] = "0"
    os.environ["SLURM_LOCALID"] = "0"
    os.environ["SLURM_NTASKS"] = "1"
    os.environ["SLURM_JOB_NODELIST"] = "localhost"
    os.environ["SLURM_JOB_NUM_NODES"] = "1"

    os.environ["RANK"] = "1"

    with pytest.warns(UserWarning, match=r"We detected the following env variables"):
        model = _NativeDistModel.create_from_backend(backend="gloo", timeout=timedelta(seconds=10))
        model.finalize()

    del os.environ["SLURM_JOB_ID"]
    del os.environ["SLURM_PROCID"]
    del os.environ["SLURM_LOCALID"]
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NODELIST"]
    del os.environ["SLURM_JOB_NUM_NODES"]
    del os.environ["RANK"]


def _assert_model(model, true_conf):
    assert model.device() == torch.device(true_conf["device"])
    assert model.get_local_rank() == true_conf["local_rank"]
    assert model.get_rank() == true_conf["rank"]
    assert model.get_world_size() == true_conf["world_size"]

    assert model.get_node_rank() == true_conf["node_index"]
    assert model.get_nnodes() == true_conf["nnodes"]
    assert model.get_nproc_per_node() == true_conf["nproc_per_node"]


def _test__native_dist_model_create_from_backend_no_dist(backend, true_device):
    from datetime import timedelta

    model = _NativeDistModel.create_from_backend(backend=backend, timeout=timedelta(seconds=20))

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

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


def _test__native_dist_model_create_from_backend_dist(init_method, local_rank, rank, world_size, backend, true_device):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)
    os.environ["RANK"] = f"{rank}"

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ

    model = _NativeDistModel.create_from_backend(backend=backend, timeout=timeout, init_method=init_method)

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    with pytest.raises(RuntimeError, match=r"Can not create new distributed process group if default one is"):
        _NativeDistModel.create_from_backend(backend=backend, timeout=timeout)

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "node_index": 0,
            "nnodes": 1,
            "nproc_per_node": world_size,
        },
    )

    if init_method is None:
        assert model._init_method == "env://"
    else:
        assert model._init_method == init_method

    model.finalize()

    del os.environ["RANK"]

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ
    assert "RANK" not in os.environ


def _test__native_dist_model_create_from_backend_slurm(local_rank, rank, world_size, backend, true_device):
    import os
    from datetime import timedelta

    timeout = timedelta(seconds=20)

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ

    del os.environ["WORLD_SIZE"]
    del os.environ["LOCAL_RANK"]

    os.environ["SLURM_JOB_ID"] = "15000"
    os.environ["SLURM_PROCID"] = str(rank)
    os.environ["SLURM_LOCALID"] = str(local_rank)
    os.environ["SLURM_NTASKS"] = str(world_size)
    os.environ["SLURM_JOB_NODELIST"] = "localhost"
    os.environ["SLURM_JOB_NUM_NODES"] = "1"

    model = _NativeDistModel.create_from_backend(backend=backend, timeout=timeout)

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    with pytest.raises(RuntimeError, match=r"Can not create new distributed process group if default one is"):
        _NativeDistModel.create_from_backend(backend=backend, timeout=timeout)

    _assert_model(
        model,
        {
            "device": true_device,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "node_index": 0,
            "nnodes": 1,
            "nproc_per_node": world_size,
        },
    )

    model.finalize()

    del os.environ["SLURM_JOB_ID"]
    del os.environ["SLURM_PROCID"]
    del os.environ["SLURM_LOCALID"]
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NODELIST"]
    del os.environ["SLURM_JOB_NUM_NODES"]

    assert "MASTER_ADDR" not in os.environ
    assert "MASTER_PORT" not in os.environ
    assert "RANK" not in os.environ

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)


def _test__native_dist_model_create_from_context_no_local_rank():
    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]

    from ignite.distributed.comp_models.base import ComputationModel

    if ComputationModel._ext_local_rank is not None:
        ComputationModel._ext_local_rank = None

    with pytest.warns(UserWarning, match=r"Local rank information for native distributed setting will be initialized"):
        _NativeDistModel.create_from_context()


def _test__native_dist_model_create_from_context_env_local_rank(true_conf):
    import os

    remove_lrank = False
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(true_conf["local_rank"])
        remove_lrank = True

    model = _NativeDistModel.create_from_context()
    _assert_model(model, true_conf)

    if remove_lrank:
        del os.environ["LOCAL_RANK"]


def _test__native_dist_model_create_from_context_set_local_rank(true_conf):
    from ignite.distributed.comp_models.base import ComputationModel

    lrank = None
    if "LOCAL_RANK" in os.environ:
        lrank = os.environ["LOCAL_RANK"]
        del os.environ["LOCAL_RANK"]

    ComputationModel._ext_local_rank = true_conf["local_rank"]

    model = _NativeDistModel.create_from_context()
    _assert_model(model, true_conf)

    ComputationModel._ext_local_rank = None

    if lrank is not None:
        os.environ["LOCAL_RANK"] = lrank


def _test__native_dist_model_create_from_context_no_dist(true_backend, true_device):
    assert _NativeDistModel.create_from_context() is None

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=1, rank=0)
    dist.barrier()

    _test__native_dist_model_create_from_context_no_local_rank()

    true_conf = {
        "device": true_device,
        "local_rank": 0,
        "rank": 0,
        "world_size": 1,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": 1,
    }

    _test__native_dist_model_create_from_context_env_local_rank(true_conf)
    _test__native_dist_model_create_from_context_set_local_rank(true_conf)

    dist.destroy_process_group()


def _test__native_dist_model_create_from_context_dist(local_rank, rank, world_size, true_backend, true_device):
    assert _NativeDistModel.create_from_context() is None

    dist.init_process_group(true_backend, "tcp://0.0.0.0:2222", world_size=world_size, rank=rank)
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    true_conf = {
        "device": true_device,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "node_index": 0,
        "nnodes": 1,
        "nproc_per_node": world_size,
    }

    _test__native_dist_model_create_from_context_env_local_rank(true_conf)
    _test__native_dist_model_create_from_context_set_local_rank(true_conf)

    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
def test__native_dist_model_create_no_dist_gloo(clean_env):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test__native_dist_model_create_from_backend_no_dist("gloo", device)
    _test__native_dist_model_create_from_context_no_dist("gloo", device)


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Should be no-dist config")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_no_dist_nccl(clean_env):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _test__native_dist_model_create_from_backend_no_dist("nccl", device)
    _test__native_dist_model_create_from_context_no_dist("nccl", device)


@pytest.mark.distributed
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test__native_dist_model_create_dist_gloo_1(init_method, get_fixed_dirname, local_rank, world_size):
    if init_method == "FILE":
        init_method = f"file://{get_fixed_dirname('native_dist_model_create_dist_gloo_1')}/shared"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _test__native_dist_model_create_from_backend_dist(init_method, local_rank, local_rank, world_size, "gloo", device)

    if init_method is None:
        _test__native_dist_model_create_from_backend_slurm(local_rank, local_rank, world_size, "gloo", device)


@pytest.mark.distributed
def test__native_dist_model_create_dist_gloo_2(local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _test__native_dist_model_create_from_context_dist(local_rank, local_rank, world_size, "gloo", device)
    _test__native_dist_model_create_from_backend_slurm(local_rank, local_rank, world_size, "gloo", device)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:22334", "FILE"])
def test__native_dist_model_create_dist_nccl_1(init_method, get_fixed_dirname, local_rank, world_size):
    if init_method == "FILE":
        init_method = f"file://{get_fixed_dirname('native_dist_model_create_dist_nccl_1')}/shared"

    _test__native_dist_model_create_from_backend_dist(
        init_method, local_rank, local_rank, world_size, "nccl", f"cuda:{local_rank}"
    )

    if init_method is None:
        _test__native_dist_model_create_from_backend_slurm(
            local_rank, local_rank, world_size, "nccl", f"cuda:{local_rank}"
        )


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test__native_dist_model_create_dist_nccl_2(local_rank, world_size):
    _test__native_dist_model_create_from_context_dist(local_rank, local_rank, world_size, "nccl", f"cuda:{local_rank}")


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Skip if less than 2 GPUs")
def test__native_dist_model_warning_index_less_localrank(local_rank, world_size):
    assert _NativeDistModel.create_from_context() is None

    dist.init_process_group("nccl", "tcp://0.0.0.0:2222", world_size=world_size, rank=local_rank)
    dist.barrier()
    # We deliberately incorrectly set cuda device to 0
    torch.cuda.set_device(0)

    model = _NativeDistModel.create_from_context()
    assert isinstance(model, _NativeDistModel), f"{type(model)} vs _NativeDistModel"

    if local_rank == 1:
        with pytest.warns(UserWarning, match=r"Current device index is less than current local rank."):
            model.device()

    dist.destroy_process_group()


def _test_dist_spawn_fn(local_rank, backend, world_size, device, **kwargs):
    from ignite.distributed.utils import _model

    assert dist.is_available() and dist.is_initialized()
    assert dist.get_backend() == backend

    assert isinstance(_model, _NativeDistModel), f"{type(_model)} vs _NativeDistModel"

    assert _model.get_local_rank() == local_rank
    assert _model.get_world_size() == world_size
    assert _model.device().type == torch.device(device).type

    if "master_addr" in kwargs:
        assert os.environ["MASTER_ADDR"] == kwargs["master_addr"]
    if "master_port" in kwargs:
        assert os.environ["MASTER_PORT"] == str(kwargs["master_port"])


def _test__native_dist_model_spawn(backend, num_workers_per_machine, device, init_method=None, **spawn_kwargs):
    kwargs_dict = {}
    for key in ["master_addr", "master_port"]:
        if key in spawn_kwargs:
            kwargs_dict[key] = spawn_kwargs[key]

    _NativeDistModel.spawn(
        _test_dist_spawn_fn,
        args=(backend, num_workers_per_machine, device),
        kwargs_dict=kwargs_dict,
        backend=backend,
        nproc_per_node=num_workers_per_machine,
        init_method=init_method,
        **spawn_kwargs,
    )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.parametrize("init_method", [None, "CUSTOM_ADDR_PORT", "env://", "tcp://0.0.0.0:22334", "FILE"])
def test__native_dist_model_spawn_gloo(init_method, dirname):
    spawn_kwargs = {}

    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"
    elif init_method == "CUSTOM_ADDR_PORT":
        init_method = None
        spawn_kwargs["master_addr"] = "0.0.0.0"
        spawn_kwargs["master_port"] = 2345

    nproc = torch.cuda.device_count() if torch.cuda.is_available() else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _test__native_dist_model_spawn(
        "gloo", num_workers_per_machine=nproc, device=device, init_method=init_method, **spawn_kwargs
    )
    if device.type == "cpu":
        spawn_kwargs["start_method"] = "fork"
        _test__native_dist_model_spawn(
            "gloo", num_workers_per_machine=nproc, device=device, init_method=init_method, **spawn_kwargs
        )

    if init_method not in [None, "env://"]:
        with pytest.raises(ValueError, match=r"master_addr should be None if init_method is provided"):
            _test__native_dist_model_spawn(
                "gloo", num_workers_per_machine=nproc, device=device, init_method=init_method, master_addr="abc"
            )
        with pytest.raises(ValueError, match=r"master_port should be None if init_method is provided"):
            _test__native_dist_model_spawn(
                "gloo", num_workers_per_machine=nproc, device=device, init_method=init_method, master_port=123
            )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
@pytest.mark.parametrize("init_method", [None, "CUSTOM_ADDR_PORT", "tcp://0.0.0.0:22334", "FILE"])
def test__native_dist_model_spawn_nccl(init_method, dirname):
    spawn_kwargs = {}

    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"
    elif init_method == "CUSTOM_ADDR_PORT":
        init_method = None
        spawn_kwargs["master_addr"] = "0.0.0.0"
        spawn_kwargs["master_port"] = 2345

    nproc = torch.cuda.device_count()
    _test__native_dist_model_spawn(
        "nccl", num_workers_per_machine=nproc, device="cuda", init_method=init_method, **spawn_kwargs
    )


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test__native_dist_model_init_method_is_none(world_size):
    with pytest.raises(ValueError, match=r"Arguments rank and world_size should be None"):
        _NativeDistModel.create_from_backend(backend="gloo", world_size=world_size)


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test__native_dist_model_init_method_is_not_none(world_size, local_rank, get_fixed_dirname):
    init_method = f"file://{get_fixed_dirname('native_dist_model_init_method_is_not_none')}/shared"
    with pytest.raises(ValueError, match=r"Both rank and world_size should be provided"):
        _NativeDistModel.create_from_backend(backend="gloo", world_size=world_size, init_method=init_method)

    with pytest.raises(ValueError, match=r"Both rank and world_size should be provided"):
        _NativeDistModel.create_from_backend(backend="gloo", rank=local_rank, init_method=init_method)


@pytest.mark.parametrize(
    "environ, expected",
    [
        # fmt: off
        # usual SLURM env
        (

            {
                "SLURM_PROCID": "1", "SLURM_LOCALID": "1", "SLURM_NTASKS": "2", "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "c1", "SLURM_JOB_ID": "12345",
            },
            [1, 1, 2, "c1", 17345]
        ),
        # usual SLURM env mnode
        (
            {
                "SLURM_PROCID": "5", "SLURM_LOCALID": "1", "SLURM_NTASKS": "8", "SLURM_JOB_NUM_NODES": "2",
                "SLURM_JOB_NODELIST": "c1, c2", "SLURM_JOB_ID": "12345",
            },
            [5, 1, 8, "c1", 17345]
        ),
        # usual SLURM env 1 node, 1 task + torch.distributed.launch
        (
            {
                "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "c1", "SLURM_JOB_ID": "12345",
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "2233", "RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "8",
            },
            [2, 2, 8, "127.0.0.1", 2233]
        ),
        # usual SLURM env + enroot's pytorch hook
        (
            {
                "SLURM_PROCID": "3", "SLURM_LOCALID": "3", "SLURM_NTASKS": "4", "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_NODELIST": "c1", "SLURM_JOB_ID": "12345",
                "MASTER_ADDR": "c1", "MASTER_PORT": "12233", "RANK": "3", "LOCAL_RANK": "3", "WORLD_SIZE": "4",
            },
            [3, 3, 4, "c1", 12233]
        ),
        # usual SLURM env mnode + enroot's pytorch hook
        (
            {
                "SLURM_PROCID": "3", "SLURM_LOCALID": "1", "SLURM_NTASKS": "4", "SLURM_JOB_NUM_NODES": "2",
                "SLURM_JOB_NODELIST": "c1, c2", "SLURM_JOB_ID": "12345",
                "MASTER_ADDR": "c1", "MASTER_PORT": "12233", "RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "4"
            },
            [3, 1, 4, "c1", 12233]
        ),
        # fmt: on
    ],
)
def test__setup_ddp_vars_from_slurm_env(environ, expected):
    ddp_keys = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    ddp_vars = _setup_ddp_vars_from_slurm_env(environ)
    for key, value in zip(ddp_keys, expected):
        assert key in ddp_vars
        assert ddp_vars[key] == value


def test__setup_ddp_vars_from_slurm_env_bad_configs():
    with pytest.raises(
        RuntimeError, match=r"Environment variable defined for PyTorch Distributed context is inconsistent"
    ):
        environ = {
            "SLURM_PROCID": "3",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_JOB_NODELIST": "c1, c2",
            "SLURM_JOB_ID": "12345",
            "MASTER_ADDR": "another-addr",
            "MASTER_PORT": "12233",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
        }
        _setup_ddp_vars_from_slurm_env(environ)

    with pytest.raises(
        RuntimeError, match=r"Environment variable defined for PyTorch Distributed context is inconsistent"
    ):
        environ = {
            "SLURM_PROCID": "1",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_JOB_NODELIST": "c1",
            "SLURM_JOB_ID": "12345",
            "MASTER_ADDR": "another-addr",
            "MASTER_PORT": "12233",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
        }
        _setup_ddp_vars_from_slurm_env(environ)

    with pytest.warns(UserWarning, match=r"We detected the following env variables"):
        environ = {
            "SLURM_PROCID": "3",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_JOB_NODELIST": "c1, c2",
            "SLURM_JOB_ID": "12345",
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
        }
        _setup_ddp_vars_from_slurm_env(environ)

    with pytest.raises(RuntimeError, match=r"No hostname detected in SLURM_JOB_NODELIST by ignite"):
        environ = {
            "SLURM_PROCID": "1",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_JOB_NODELIST": "[]",
            "SLURM_JOB_ID": "12345",
        }
        _setup_ddp_vars_from_slurm_env(environ)
