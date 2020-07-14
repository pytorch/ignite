import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import ignite.distributed as idist
from ignite.distributed.utils import has_native_dist_support, has_xla_support


@pytest.fixture()
def exec_filepath():
    fp = Path(__file__).parent / "check_idist_parallel.py"
    assert fp.exists()
    yield fp.as_posix()


def execute(cmd):

    import ignite

    env = dict(os.environ)
    env["PYTHONPATH"] = "{}".format(os.path.dirname(ignite.__path__[0]))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    process.wait()
    if process.returncode != 0:
        print(str(process.stdout.read()) + str(process.stderr.read()))
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd, stderr=process.stderr.read())
    return str(process.stdout.read()) + str(process.stderr.read())


def test_check_idist_parallel_no_dist(exec_filepath):
    cmd = [sys.executable, "-u", exec_filepath]
    out = execute(cmd)
    assert "backend=None" in out
    assert "in 1 processes" in out
    assert "End of run" in out


def _test_check_idist_parallel_torch_launch(fp, backend, nprocs):
    # python -m torch.distributed.launch --nproc_per_node=nprocs --use_env \
    #   tests/ignite/distributed/check_idist_parallel.py --backend=backend

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node={}".format(nprocs),
        "--use_env",
        fp,
        "--backend={}".format(backend),
    ]

    out = execute(cmd)
    assert "backend={}".format(backend) in out
    assert "in {} processes".format(nprocs) in out
    assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test runs torch launch")
def test_check_idist_parallel_torch_launch_n_procs_gloo(exec_filepath):
    _test_check_idist_parallel_torch_launch(exec_filepath, "gloo", 4)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test runs torch launch")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_check_idist_parallel_torch_launch_n_procs_nccl(exec_filepath):
    _test_check_idist_parallel_torch_launch(exec_filepath, "nccl", torch.cuda.device_count())


def _test_check_idist_parallel_spawn(fp, backend, nprocs):
    # python tests/ignite/distributed/check_idist_parallel.py --backend=backend --nproc_per_node=nprocs

    cmd = [sys.executable, fp, "--backend={}".format(backend), "--nproc_per_node={}".format(nprocs)]

    out = execute(cmd)
    assert "backend={}".format(backend) in out
    assert "Spawn function" in out
    assert "in {} processes".format(nprocs) in out
    if "xla" not in backend:
        assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_check_idist_parallel_spawn_n_procs_gloo(exec_filepath):
    _test_check_idist_parallel_spawn(exec_filepath, "gloo", 4)


@pytest.mark.distributed
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_check_idist_parallel_spawn_n_procs_nccl(exec_filepath):
    _test_check_idist_parallel_spawn(exec_filepath, "nccl", torch.cuda.device_count())


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_check_idist_parallel_spawn_n_procs_xla(exec_filepath):
    n = int(os.environ["NUM_TPU_WORKERS"])
    if n > 1:
        _test_check_idist_parallel_spawn(exec_filepath, "xla-tpu", n)


def _test_func(index, ws, device):
    assert 0 <= index < ws
    assert ws == idist.get_world_size()
    assert device in idist.device().type


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_parallel_gloo():
    with idist.Parallel(backend="gloo", nproc_per_node=4) as parallel:
        parallel.run(_test_func, ws=4, device="cpu")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Skip if not launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
def test_idist_parallel_gloo_nprocs(local_rank, world_size):
    os.environ["RANK"] = str(local_rank)
    with idist.Parallel(backend="gloo") as parallel:
        parallel.run(_test_func, ws=world_size, device="cpu")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_parallel_nccl():
    with idist.Parallel(backend="nccl", nproc_per_node=torch.cuda.device_count()) as parallel:
        parallel.run(_test_func, ws=torch.cuda.device_count(), device="cuda")


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" not in os.environ, reason="Skip if not launched as multiproc")
@pytest.mark.skipif(not has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_idist_parallel_nccl_nprocs(local_rank, world_size):
    os.environ["RANK"] = str(local_rank)
    with idist.Parallel(backend="nccl") as parallel:
        parallel.run(_test_func, ws=world_size, device="cuda")


@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_idist_parallel_no_dist():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with idist.Parallel(backend=None) as parallel:
        parallel.run(_test_func, ws=1, device=device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_parallel_spawn_params():

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
