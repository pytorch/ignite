import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ignite.distributed.utils import has_xla_support


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
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test runs torch launch")
def test_check_idist_parallel_torch_launch_n_procs_gloo(exec_filepath):
    _test_check_idist_parallel_torch_launch(exec_filepath, "gloo", 4)


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip because test runs torch launch")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_check_idist_parallel_torch_launch_n_procs_nccl(exec_filepath):
    _test_check_idist_parallel_torch_launch(exec_filepath, "nccl", torch.cuda.device_count())


def _test_check_idist_parallel_spawn(fp, backend, nprocs):
    # python tests/ignite/distributed/check_idist_parallel.py --backend=backend --num_procs_per_node=nprocs

    cmd = [sys.executable, fp, "--backend={}".format(backend), "--num_procs_per_node={}".format(nprocs)]

    out = execute(cmd)
    assert "backend={}".format(backend) in out
    assert "Spawn function" in out
    assert "in {} processes".format(nprocs) in out
    if "xla" not in backend:
        assert "End of run" in out


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_check_idist_parallel_spawn_n_procs_gloo(exec_filepath):
    _test_check_idist_parallel_spawn(exec_filepath, "gloo", 4)


@pytest.mark.distributed
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_check_idist_parallel_spawn_n_procs_nccl(exec_filepath):
    _test_check_idist_parallel_spawn(exec_filepath, "nccl", torch.cuda.device_count())


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_check_idist_parallel_spawn_n_procs_xla(exec_filepath):
    n = int(os.environ["NUM_TPU_WORKERS"])
    _test_check_idist_parallel_spawn(exec_filepath, "xla-tpu", n)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not has_xla_support, reason="Skip if no PyTorch XLA package")
def test_idist_methods_in_xla_context(exec_filepath):
    _test_check_idist_parallel_spawn(exec_filepath, "xla-tpu", 1)
