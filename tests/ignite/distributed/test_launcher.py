import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from packaging.version import Version

import ignite.distributed as idist
from ignite.distributed.utils import has_hvd_support, has_native_dist_support, has_xla_support
from tests.ignite import is_mps_available_and_functional


def test_parallel_wrong_inputs():
    with pytest.raises(ValueError, match=r"Unknown backend 'abc'. Available backends:"):
        idist.Parallel(backend="abc")

    with pytest.raises(ValueError, match=r"If backend is None, argument 'nnodes' should be also None"):
        idist.Parallel(nnodes=2)

    with pytest.raises(ValueError, match=r"Argument nproc_per_node should positive"):
        idist.Parallel(backend="gloo", nproc_per_node=-1)

    with pytest.raises(ValueError, match=r"Argument nnodes should positive"):
        idist.Parallel(backend="gloo", nproc_per_node=1, nnodes=-1)

    with pytest.raises(ValueError, match=r"If number of nodes larger than one"):
        idist.Parallel(backend="gloo", nproc_per_node=1, nnodes=2)

    with pytest.raises(ValueError, match=r"Argument node_rank should be between 0 and"):
        idist.Parallel(backend="gloo", nproc_per_node=1, nnodes=2, node_rank=2)

    with pytest.raises(ValueError, match=r"If number of nodes larger than one, arguments master_addr and master_port"):
        idist.Parallel(backend="gloo", nproc_per_node=1, nnodes=2, node_rank=1)


@pytest.fixture()
def exec_filepath():
    fp = Path(__file__).parent / "check_idist_parallel.py"
    assert fp.exists()
    yield fp.as_posix()


def execute(cmd, env=None):
    import ignite

    env = dict(os.environ) if env is None else env
    env["PYTHONPATH"] = f"{os.path.dirname(ignite.__path__[0])}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    process.wait()
    if process.returncode != 0:
        print(str(process.stdout.read()) + str(process.stderr.read()))
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd, stderr=process.stderr.read())
    return str(process.stdout.read()) + str(process.stderr.read())


@pytest.mark.skipif(not is_mps_available_and_functional(), reason="Skip if MPS not functional")
def test_check_idist_parallel_no_dist(exec_filepath):
    cmd = [sys.executable, "-u", exec_filepath]
    out = execute(cmd)
    assert "backend=None" in out
    assert "in 1 processes" in out
    assert "End of run" in out


def _test_check_idist_parallel_torch_launch(init_method, fp, backend, nprocs):
    # torchrun --nproc_per_node=nprocs tests/ignite/distributed/check_idist_parallel.py --backend=backend

    cmd = []
    if Version(torch.__version__) >= Version("1.10.0"):
        cmd += ["torchrun"]
    else:
        cmd += [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--use_env",
        ]

    cmd += [
        f"--nproc_per_node={nprocs}",
        fp,
        f"--backend={backend}",
    ]

    if init_method is not None:
        cmd.append(f"--init_method={init_method}")

    out = execute(cmd)
    assert f"backend={backend}" in out
    assert f"in {nprocs} processes" in out
    assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test uses torch launch")
@pytest.mark.parametrize("init_method", [None, "tcp://0.0.0.0:29500", "FILE"])
@pytest.mark.parametrize(
    "backend",
    ["gloo", pytest.param("nccl", marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU"))],
)
def test_check_idist_parallel_torch_launch_n_procs_native(init_method, dirname, exec_filepath, backend):
    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"

    np = torch.cuda.device_count() if torch.cuda.is_available() else 4
    _test_check_idist_parallel_torch_launch(init_method, exec_filepath, backend, np)


def _test_check_idist_parallel_hvdrun(fp, backend, nprocs):
    # horovodrun -np=nprocs python tests/ignite/distributed/check_idist_parallel.py --backend=backend

    cmd = [
        "horovodrun",
        "-np",
        f"{nprocs}",
        sys.executable,
        fp,
        f"--backend={backend}",
    ]

    out = execute(cmd)
    assert f"backend={backend}" in out
    assert f"in {nprocs} processes" in out
    assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test uses horovodrun")
def test_check_idist_parallel_hvdrun_launch_n_procs(exec_filepath):
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    _test_check_idist_parallel_hvdrun(exec_filepath, "horovod", np)


def _test_check_idist_parallel_spawn(fp, backend, nprocs):
    # python tests/ignite/distributed/check_idist_parallel.py --backend=backend --nproc_per_node=nprocs

    cmd = [sys.executable, fp, f"--backend={backend}", f"--nproc_per_node={nprocs}"]

    out = execute(cmd)
    assert f"backend={backend}" in out
    assert "Spawn function" in out
    assert f"in {nprocs} processes" in out
    if "xla" not in backend:
        assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.parametrize(
    "backend",
    ["gloo", pytest.param("nccl", marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU"))],
)
def test_check_idist_parallel_spawn_n_procs_native(exec_filepath, backend):
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    _test_check_idist_parallel_spawn(exec_filepath, backend, np)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_smoke_test_check_idist_parallel_spawn_multinode_n_procs_gloo(exec_filepath):
    # Just a smoke test from check_idist_parallel.py for an emulated multi-node configuration
    cmd1 = "export CUDA_VISIBLE_DEVICES= && "
    cmd1 += f'bash -c "{sys.executable} {exec_filepath} --backend=gloo --nproc_per_node=2 '
    cmd1 += '--nnodes=2 --node_rank=0 --master_addr=localhost --master_port=3344 &"'
    os.system(cmd1)

    cmd2 = [
        sys.executable,
        exec_filepath,
        "--backend=gloo",
        "--nproc_per_node=2",
        "--nnodes=2",
        "--node_rank=1",
        "--master_addr=localhost",
        "--master_port=3344",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = execute(cmd2, env=env)

    assert "backend=gloo" in out
    assert "nproc_per_node: 2" in out
    assert "nnodes: 2" in out
    assert "master_addr: localhost" in out
    assert "master_port: 3344" in out
    assert "End of run" in out


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_check_idist_parallel_spawn_n_procs_xla(exec_filepath):
    n = int(os.environ["NUM_TPU_WORKERS"])
    if n > 1:
        _test_check_idist_parallel_spawn(exec_filepath, "xla-tpu", n)


@pytest.mark.distributed
@pytest.mark.skipif(not has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_check_idist_parallel_spawn_n_procs_hvd(exec_filepath):
    np = 4 if not torch.cuda.is_available() else torch.cuda.device_count()
    _test_check_idist_parallel_spawn(exec_filepath, "horovod", np)


def _test_func(index, ws, device, backend, true_init_method):
    assert 0 <= index < ws
    assert index == idist.get_local_rank()
    assert ws == idist.get_world_size()
    assert torch.device(device).type == idist.device().type
    assert backend == idist.backend()

    if idist.model_name() == "native-dist":
        from ignite.distributed.utils import _model

        assert _model._init_method == true_init_method


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.parametrize("init_method", ["env://", "tcp://0.0.0.0:29500", "FILE"])
@pytest.mark.parametrize(
    "backend",
    ["gloo", pytest.param("nccl", marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU"))],
)
def test_idist_parallel_spawn_n_procs_native(init_method, backend, dirname):
    if init_method == "FILE":
        init_method = f"file://{dirname}/shared"

    nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node, init_method=init_method) as parallel:
        parallel.run(_test_func, ws=nproc_per_node, device=device, backend=backend, true_init_method=init_method)


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Skip if not launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.parametrize("init_method", ["env://", "tcp://0.0.0.0:29500", "FILE"])
@pytest.mark.parametrize(
    "backend",
    ["gloo", pytest.param("nccl", marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU"))],
)
def test_idist_parallel_n_procs_native(init_method, backend, get_fixed_dirname, local_rank, world_size):
    if init_method == "FILE":
        init_method = f"file://{get_fixed_dirname('idist_parallel_n_procs_native')}/shared"

    os.environ["RANK"] = str(local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with idist.Parallel(backend=backend, init_method=init_method) as parallel:
        parallel.run(_test_func, ws=world_size, device=device, backend=backend, true_init_method=init_method)


@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_parallel_no_dist():
    device = idist.device()
    with idist.Parallel(backend=None) as parallel:
        parallel.run(_test_func, ws=1, device=device, backend=None, true_init_method=None)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_parallel_spawn_params_xla():
    res = idist.Parallel._setup_spawn_params(
        nproc_per_node=8, nnodes=None, node_rank=None, master_addr=None, master_port=None, start_method="fork"
    )
    assert "nproc_per_node" in res and res["nproc_per_node"] == 8
    assert "start_method" in res and res["start_method"] == "fork"

    with idist.Parallel(backend="xla-tpu", nproc_per_node=8, start_method="fork") as parallel:
        assert parallel.backend == "xla-tpu"
        res = parallel._spawn_params
        assert "nproc_per_node" in res and res["nproc_per_node"] == 8
        assert "start_method" in res and res["start_method"] == "fork"
