"""
Distributed smoke test for FSDP wrapping via auto_model.
Must be run as a standalone script (not via pytest) using:
  python tests/ignite/test_fsdp_distributed.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_worker(rank: int, world_size: int, backend: str, results: dict) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    try:
        from ignite.distributed.auto import auto_model
        from torch.distributed._composable.fsdp import FSDPModule

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Test 1: use_fsdp=True applies FSDP2 when world_size > 1
        model = nn.Linear(10, 10).to(device)
        wrapped = auto_model(model, use_fsdp=True)
        assert isinstance(wrapped, FSDPModule), f"[rank {rank}] Expected FSDPModule, got {type(wrapped).__name__}"
        results[f"rank{rank}_fsdp_wrap"] = True

        # Test 2: use_fsdp=True + sync_bn=True raises ValueError (all ranks)
        try:
            auto_model(nn.Linear(5, 5), use_fsdp=True, sync_bn=True)
            results[f"rank{rank}_valueerror"] = False  # Should not reach here
        except ValueError:
            results[f"rank{rank}_valueerror"] = True

        # Test 3: forward pass through FSDP-wrapped model works
        x = torch.randn(4, 10, device=device)
        out = wrapped(x)
        assert out.shape == (4, 10), f"[rank {rank}] Unexpected output shape: {out.shape}"
        results[f"rank{rank}_forward"] = True

        dist.barrier()

    except Exception as e:
        results[f"rank{rank}_error"] = str(e)
        raise
    finally:
        dist.destroy_process_group()


def run_checkpoint_worker(rank: int, world_size: int, backend: str, tmpdir: str, results: dict) -> None:
    """Test FSDP checkpoint save/load in distributed context."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12400"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        from ignite.handlers import Checkpoint, DiskSaver
        from ignite.engine import Engine, Events
        from ignite.engine.engine import State
        from torch.distributed._composable.fsdp import fully_shard

        torch.manual_seed(0)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).to(device)
        fully_shard(model)

        save_dir = os.path.join(tmpdir, "fsdp_ckpt")
        os.makedirs(save_dir, exist_ok=True)
        dist.barrier()

        checkpointer = Checkpoint(
            {"model": model},
            DiskSaver(save_dir, create_dir=False, require_empty=False),
        )
        engine = Engine(lambda e, b: None)
        engine.state = State(epoch=0, iteration=0)
        checkpointer(engine)

        # Only rank 0 should have a non-empty file
        dist.barrier()

        if rank == 0:
            import pathlib
            ckpt_files = list(pathlib.Path(save_dir).glob("model_*.pt"))
            assert len(ckpt_files) == 1, f"Expected 1 checkpoint, found {ckpt_files}"
            saved = torch.load(ckpt_files[0], map_location="cpu")
            # Saved keys must match unwrapped model
            expected_keys = set(nn.Sequential(nn.Linear(8, 8), nn.ReLU()).state_dict().keys())
            assert set(saved.keys()) == expected_keys, (
                f"Checkpoint keys mismatch. Got {set(saved.keys())}, expected {expected_keys}"
            )
            results["ckpt_save"] = True
        else:
            results[f"rank{rank}_ckpt_skip"] = True

        dist.barrier()

    except Exception as e:
        results[f"rank{rank}_ckpt_error"] = str(e)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing as mp
    import tempfile

    manager = mp.Manager()
    results: dict = manager.dict()

    print("=" * 60)
    print("Test 1: auto_model FSDP wrapping (gloo, 2 processes)")
    print("=" * 60)

    processes = []
    for rank in range(2):
        p = mp.Process(target=run_worker, args=(rank, 2, "gloo", results))
        p.start()
        processes.append(p)

    exit_codes = []
    for p in processes:
        p.join()
        exit_codes.append(p.exitcode)

    if all(ec == 0 for ec in exit_codes):
        print("PASSED: auto_model FSDP wrapping with gloo")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")
    else:
        print("FAILED: auto_model FSDP wrapping with gloo")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Test 2: FSDP checkpoint save (gloo, 2 processes)")
    print("=" * 60)

    results2: dict = manager.dict()
    with tempfile.TemporaryDirectory() as tmpdir:
        processes2 = []
        for rank in range(2):
            p = mp.Process(target=run_checkpoint_worker, args=(rank, 2, "gloo", tmpdir, results2))
            p.start()
            processes2.append(p)

        exit_codes2 = []
        for p in processes2:
            p.join()
            exit_codes2.append(p.exitcode)

    if all(ec == 0 for ec in exit_codes2):
        print("PASSED: FSDP checkpoint save with gloo")
        for k, v in sorted(results2.items()):
            print(f"  {k}: {v}")
    else:
        print("FAILED: FSDP checkpoint save with gloo")
        for k, v in sorted(results2.items()):
            print(f"  {k}: {v}")
        sys.exit(1)

    print()
    print("All distributed FSDP tests PASSED")
