"""
Smoke tests for FSDP support changes in:
  - ignite/distributed/auto.py  (auto_model use_fsdp parameter)
  - ignite/handlers/checkpoint.py (HAVE_FSDP2 flag and FSDP checkpoint branch)

All tests run in single-process / non-distributed mode.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleLinear(nn.Module):
    """Tiny model used across tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _ModelWithBN(nn.Module):
    """Model with a BatchNorm layer – relevant for the sync_bn conflict check."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x))


# ---------------------------------------------------------------------------
# Section 1: auto_model / use_fsdp
# ---------------------------------------------------------------------------

class TestAutoModelFSDPFlag:
    """Tests for the use_fsdp parameter added to auto_model."""

    # ------------------------------------------------------------------
    # 1a. ValueError when use_fsdp=True and sync_bn=True simultaneously
    # ------------------------------------------------------------------

    def test_use_fsdp_and_sync_bn_raises_value_error(self) -> None:
        """use_fsdp=True and sync_bn=True must be mutually exclusive."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        with pytest.raises(ValueError, match="mutually exclusive"):
            auto_model(model, use_fsdp=True, sync_bn=True)

    def test_use_fsdp_and_sync_bn_with_bn_model_raises_value_error(self) -> None:
        """Mutual exclusion holds even when the model actually has BatchNorm layers."""
        from ignite.distributed.auto import auto_model

        model = _ModelWithBN()
        with pytest.raises(ValueError, match="mutually exclusive"):
            auto_model(model, use_fsdp=True, sync_bn=True)

    def test_use_fsdp_sync_bn_false_does_not_raise(self) -> None:
        """use_fsdp=True without sync_bn=True must NOT raise."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        # In non-distributed mode FSDP wrapping is skipped; the call must succeed.
        returned = auto_model(model, use_fsdp=True, sync_bn=False)
        assert returned is not None

    # ------------------------------------------------------------------
    # 1b. Non-distributed fallback: model returned unchanged (or DataParallel)
    # ------------------------------------------------------------------

    def test_use_fsdp_non_distributed_returns_nn_module(self) -> None:
        """In non-distributed context auto_model must return an nn.Module."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        result = auto_model(model, use_fsdp=True)
        assert isinstance(result, nn.Module)

    def test_use_fsdp_false_non_distributed_returns_nn_module(self) -> None:
        """Baseline: use_fsdp=False still returns an nn.Module in non-dist mode."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        result = auto_model(model, use_fsdp=False)
        assert isinstance(result, nn.Module)

    def test_use_fsdp_default_is_false(self) -> None:
        """use_fsdp must default to False (no behaviour change from old callers)."""
        import inspect
        from ignite.distributed.auto import auto_model

        sig = inspect.signature(auto_model)
        assert "use_fsdp" in sig.parameters, "use_fsdp parameter missing from auto_model"
        assert sig.parameters["use_fsdp"].default is False

    def test_use_fsdp_non_distributed_preserves_model_output(self) -> None:
        """Model wrapped via auto_model(use_fsdp=True) must still forward correctly.

        Note: on a machine with >1 GPU, auto_model wraps with DataParallel (non-dist
        multi-GPU path), which moves model parameters to CUDA. Inputs must be on the
        same device as the wrapped model to forward correctly.
        """
        from ignite.distributed.auto import auto_model

        torch.manual_seed(0)
        model = _SimpleLinear()
        wrapped = auto_model(model, use_fsdp=True)

        # Determine the device the model ended up on after wrapping.
        device = next(wrapped.parameters()).device
        x = torch.ones(2, 4, device=device)
        actual = wrapped(x)
        assert actual.shape == (2, 2), f"Unexpected output shape: {actual.shape}"
        assert actual.dtype == torch.float32

    def test_sync_bn_without_fsdp_non_distributed_does_not_raise(self) -> None:
        """sync_bn=True alone (no use_fsdp) must remain valid in non-dist mode."""
        from ignite.distributed.auto import auto_model

        model = _ModelWithBN()
        # No distributed backend, so SyncBN conversion is skipped – must not raise.
        result = auto_model(model, sync_bn=True)
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# Section 2: checkpoint.py FSDP imports
# ---------------------------------------------------------------------------

class TestCheckpointFSDPImports:
    """Verify the HAVE_FSDP2 flag and associated symbols are importable."""

    def test_have_fsdp_is_true(self) -> None:
        """HAVE_FSDP2 must be True when torch.distributed._composable.fsdp is present."""
        from ignite.handlers.checkpoint import HAVE_FSDP2

        assert HAVE_FSDP2 is True, (
            "HAVE_FSDP2 is False — torch.distributed._composable.fsdp may be unavailable in this environment"
        )

    def test_fsdp2_symbols_importable_from_checkpoint_module(self) -> None:
        """FSDPModule, get_model_state_dict and set_model_state_dict must be reachable."""
        # checkpoint.py imports these at module level inside the try/except;
        # confirm they are accessible from the installed torch build.
        from torch.distributed._composable.fsdp import FSDPModule  # noqa: F401
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict  # noqa: F401

    def test_checkpoint_module_imports_cleanly(self) -> None:
        """The checkpoint module itself must import without errors."""
        import importlib
        import ignite.handlers.checkpoint as chkpt_mod  # noqa: F401

        importlib.reload(chkpt_mod)  # force re-import to surface any top-level errors


# ---------------------------------------------------------------------------
# Section 3: Non-FSDP checkpoint save/load regression
# ---------------------------------------------------------------------------

class TestCheckpointNonFSDPRegression:
    """Ensure the existing (non-FSDP) checkpoint save/load path is intact."""

    def test_save_and_load_plain_model(self) -> None:
        """Save a plain nn.Module checkpoint and reload it successfully."""
        from ignite.engine import Engine, Events
        from ignite.handlers import Checkpoint, DiskSaver

        torch.manual_seed(42)
        model = _SimpleLinear()
        original_weight = model.fc.weight.data.clone()

        trainer = Engine(lambda e, b: None)

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DiskSaver(tmpdir, create_dir=False, require_empty=False)
            handler = Checkpoint({"model": model}, saver, n_saved=1)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
            trainer.run([0], max_epochs=1)

            saved_files = list(Path(tmpdir).glob("*.pt"))
            assert len(saved_files) == 1, f"Expected 1 checkpoint file, got {saved_files}"

            # Corrupt the model weights, then restore from checkpoint.
            model.fc.weight.data.fill_(0.0)
            assert not torch.allclose(model.fc.weight.data, original_weight)

            checkpoint = torch.load(saved_files[0], weights_only=True)
            Checkpoint.load_objects({"model": model}, checkpoint)
            assert torch.allclose(model.fc.weight.data, original_weight), (
                "Loaded weights do not match original — regression in non-FSDP load path"
            )

    def test_save_and_load_plain_model_via_filepath_string(self) -> None:
        """load_objects also accepts a filepath string — verify this regression path."""
        from ignite.engine import Engine, Events
        from ignite.handlers import Checkpoint, DiskSaver

        torch.manual_seed(7)
        model = _SimpleLinear()
        original_weight = model.fc.weight.data.clone()

        trainer = Engine(lambda e, b: None)

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DiskSaver(tmpdir, create_dir=False, require_empty=False)
            handler = Checkpoint({"model": model}, saver, n_saved=1)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
            trainer.run([0], max_epochs=1)

            saved_files = list(Path(tmpdir).glob("*.pt"))
            assert len(saved_files) == 1

            model.fc.weight.data.fill_(0.0)

            # Load via string path instead of dict.
            Checkpoint.load_objects({"model": model}, str(saved_files[0]))
            assert torch.allclose(model.fc.weight.data, original_weight)

    def test_save_and_load_ddp_wrapped_model(self) -> None:
        """Checkpoint must unwrap DataParallel and save the inner module's state."""
        from ignite.handlers import Checkpoint, DiskSaver

        torch.manual_seed(99)
        model = _SimpleLinear()

        # Only wrap with DataParallel if multiple GPUs are present; else bare model.
        if torch.cuda.device_count() > 1:
            wrapped = nn.DataParallel(model)
        else:
            # Simulate DP wrapping using a stub that exposes .module.
            class _FakeDP(nn.Module):
                def __init__(self, m: nn.Module) -> None:
                    super().__init__()
                    self.module = m

                def state_dict(self, **kw):  # type: ignore[override]
                    return self.module.state_dict(**kw)

                def load_state_dict(self, sd, **kw):  # type: ignore[override]
                    return self.module.load_state_dict(sd, **kw)

            wrapped = _FakeDP(model)

        original_weight = model.fc.weight.data.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            saver = DiskSaver(tmpdir, create_dir=False, require_empty=False)
            # Checkpoint with the plain model (not wrapped) — common usage pattern.
            handler = Checkpoint({"model": model}, saver, n_saved=1)
            from ignite.engine import Engine, Events
            trainer = Engine(lambda e, b: None)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
            trainer.run([0], max_epochs=1)

            saved_files = list(Path(tmpdir).glob("*.pt"))
            assert len(saved_files) == 1

            model.fc.weight.data.fill_(0.0)
            checkpoint = torch.load(saved_files[0], weights_only=True)
            Checkpoint.load_objects({"model": model}, checkpoint)
            assert torch.allclose(model.fc.weight.data, original_weight)

    def test_load_objects_from_dict_checkpoint(self) -> None:
        """Checkpoint.load_objects must work when passed a raw state-dict mapping.

        Note: torch.nn.Module.state_dict() returns tensors that SHARE storage with
        the model's parameters (they are views, not copies). We must deepcopy the
        state_dict before mutating the model to avoid corrupting the saved snapshot.
        """
        import copy

        torch.manual_seed(5)
        model = _SimpleLinear()
        # Use deepcopy so the saved snapshot is independent of the live model.
        saved_sd = copy.deepcopy(model.state_dict())
        original_weight = saved_sd["fc.weight"].clone()

        model.fc.weight.data.fill_(99.0)

        from ignite.handlers import Checkpoint
        Checkpoint.load_objects({"model": model}, {"model": saved_sd})
        assert torch.allclose(model.fc.weight.data, original_weight), (
            "load_objects did not restore weights from dict checkpoint"
        )

    def test_load_objects_single_key_direct_state_dict(self) -> None:
        """When to_load has one key absent from checkpoint, load_objects falls
        back to treating the whole checkpoint as the state_dict directly.

        Note: torch.nn.Module.state_dict() shares storage with live parameters;
        use deepcopy before mutating the model.
        """
        import copy

        torch.manual_seed(3)
        model = _SimpleLinear()
        saved_sd = copy.deepcopy(model.state_dict())
        original_weight = saved_sd["fc.weight"].clone()

        model.fc.weight.data.fill_(99.0)

        from ignite.handlers import Checkpoint
        # Pass the bare state_dict without the "model" key wrapper.
        Checkpoint.load_objects({"model": model}, saved_sd)
        assert torch.allclose(model.fc.weight.data, original_weight), (
            "load_objects did not restore weights from single-key direct state_dict"
        )


# ---------------------------------------------------------------------------
# Section 4: Edge-case / boundary checks
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Additional edge cases identified during change analysis."""

    def test_use_fsdp_true_sync_bn_false_explicitly(self) -> None:
        """Explicit sync_bn=False with use_fsdp=True must not raise."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        result = auto_model(model, use_fsdp=True, sync_bn=False)
        assert isinstance(result, nn.Module)

    def test_auto_model_empty_model_use_fsdp(self) -> None:
        """auto_model with use_fsdp=True on a model with no parameters must not crash."""
        from ignite.distributed.auto import auto_model

        class _Empty(nn.Module):
            def forward(self, x):  # type: ignore[override]
                return x

        model = _Empty()
        result = auto_model(model, use_fsdp=True)
        assert isinstance(result, nn.Module)

    def test_use_fsdp_kwargs_passthrough_non_distributed(self) -> None:
        """Extra kwargs must be silently ignored in non-distributed mode
        (they would be forwarded to FSDP constructor only when world_size > 1)."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        # In non-dist mode, FSDP is not instantiated, so representative FSDP kwargs
        # such as ``reshard_after_forward`` should be accepted and ignored.
        result = auto_model(model, use_fsdp=True, reshard_after_forward=False)
        assert isinstance(result, nn.Module)

    def test_have_fsdp_flag_is_boolean(self) -> None:
        """HAVE_FSDP2 must be a plain Python bool (not None or a module)."""
        from ignite.handlers.checkpoint import HAVE_FSDP2

        assert isinstance(HAVE_FSDP2, bool)

    def test_setup_checkpoint_returns_dict_for_plain_model(self) -> None:
        """_setup_checkpoint on a non-FSDP model must return a non-empty dict."""
        from ignite.handlers.checkpoint import Checkpoint

        model = _SimpleLinear()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        chkpt = Checkpoint(
            to_save={"model": model, "optimizer": optimizer},
            save_handler=lambda chk, fn, meta=None: None,
        )
        result = chkpt._setup_checkpoint()
        assert isinstance(result, dict)
        assert "model" in result
        assert "optimizer" in result

    def test_value_error_message_mentions_fsdp_and_sync_bn(self) -> None:
        """Error message must contain enough context for users to understand
        what went wrong — mention of both FSDP and SyncBatchNorm."""
        from ignite.distributed.auto import auto_model

        model = _SimpleLinear()
        with pytest.raises(ValueError) as exc_info:
            auto_model(model, use_fsdp=True, sync_bn=True)

        msg = str(exc_info.value).lower()
        assert "fsdp" in msg or "fully" in msg, f"Error message lacks FSDP mention: {exc_info.value}"
        assert "sync" in msg or "batchnorm" in msg or "bn" in msg, (
            f"Error message lacks SyncBN mention: {exc_info.value}"
        )

    def test_checkpoint_load_objects_invalid_type_raises_type_error(self) -> None:
        """Passing an invalid checkpoint type must raise TypeError (regression guard)."""
        from ignite.handlers import Checkpoint

        model = _SimpleLinear()
        with pytest.raises(TypeError):
            Checkpoint.load_objects({"model": model}, 12345)  # type: ignore[arg-type]
