import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

import ignite.handlers as handlers
from ignite.contrib.engines.common import setup_any_logging
from ignite.engine import Engine, Events
from ignite.handlers.logger_utils import (
    _setup_logging,
    setup_clearml_logging,
    setup_mlflow_logging,
    setup_neptune_logging,
    setup_plx_logging,
    setup_tb_logging,
    setup_trains_logging,
    setup_visdom_logging,
    setup_wandb_logging,
)


def _test_setup_logging(
    setup_logging_fn,
    kwargs_dict,
    output_handler_cls,
    opt_params_handler_cls,
    with_eval=True,
    with_optim=True,
    as_class=False,
    log_every_iters=1,
):
    trainer = Engine(lambda e, b: b)
    evaluators = None
    optimizers = None

    if with_eval:
        evaluator = Engine(lambda e, b: None)
        acc_scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.0]

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine):
            evaluator.run([0, 1])

        @evaluator.on(Events.EPOCH_COMPLETED)
        def set_eval_metric(engine):
            engine.state.metrics = {"acc": acc_scores[trainer.state.epoch - 1]}

        evaluators = {"validation": evaluator}
        if as_class:
            evaluators = evaluators["validation"]

    if with_optim:
        t = torch.tensor([0])
        optimizers = {"optimizer": torch.optim.SGD([t], lr=0.01)}
        if as_class:
            optimizers = optimizers["optimizer"]

    kwargs_dict["trainer"] = trainer
    kwargs_dict["optimizers"] = optimizers
    kwargs_dict["evaluators"] = evaluators
    kwargs_dict["log_every_iters"] = log_every_iters

    x_logger = setup_logging_fn(**kwargs_dict)

    handlers = trainer._event_handlers[Events.ITERATION_COMPLETED]
    for cls in [
        output_handler_cls,
    ]:
        assert any([isinstance(h[0], cls) for h in handlers]), f"{handlers}"

    if with_optim:
        handlers = trainer._event_handlers[Events.ITERATION_STARTED]
        for cls in [
            opt_params_handler_cls,
        ]:
            assert any([isinstance(h[0], cls) for h in handlers]), f"{handlers}"

    if with_eval:
        handlers = evaluator._event_handlers[Events.COMPLETED]
        for cls in [
            output_handler_cls,
        ]:
            assert any([isinstance(h[0], cls) for h in handlers]), f"{handlers}"

    data = [0, 1, 2]
    trainer.run(data, max_epochs=10)

    if "output_path" in kwargs_dict:
        tb_files = list(os.listdir(kwargs_dict["output_path"]))
        assert len(tb_files) == 1
        for v in [
            "events",
        ]:
            assert any([v in c for c in tb_files]), f"{tb_files}"

    return x_logger


def test_deprecated_setup_any_logging():
    with pytest.raises(DeprecationWarning, match=r"deprecated since version 0.4.0"):
        setup_any_logging(None, None, None, None, None, None)


def test__setup_logging_wrong_args():
    with pytest.raises(TypeError, match=r"Argument optimizers should be either a single optimizer or"):
        _setup_logging(MagicMock(), MagicMock(), "abc", MagicMock(), 1)

    with pytest.raises(TypeError, match=r"Argument evaluators should be either a single engine or"):
        _setup_logging(MagicMock(), MagicMock(), MagicMock(spec=torch.optim.SGD), "abc", 1)


def test_setup_tb_logging(dirname):
    tb_logger = _test_setup_logging(
        setup_logging_fn=setup_tb_logging,
        kwargs_dict={"output_path": dirname / "t1"},
        output_handler_cls=handlers.tensorboard_logger.OutputHandler,
        opt_params_handler_cls=handlers.tensorboard_logger.OptimizerParamsHandler,
        with_eval=False,
        with_optim=False,
    )
    tb_logger.close()
    tb_logger = _test_setup_logging(
        setup_logging_fn=setup_tb_logging,
        kwargs_dict={"output_path": dirname / "t2"},
        output_handler_cls=handlers.tensorboard_logger.OutputHandler,
        opt_params_handler_cls=handlers.tensorboard_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
    )
    tb_logger.close()
    tb_logger = _test_setup_logging(
        setup_logging_fn=setup_tb_logging,
        kwargs_dict={"output_path": dirname / "t3"},
        output_handler_cls=handlers.tensorboard_logger.OutputHandler,
        opt_params_handler_cls=handlers.tensorboard_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
        as_class=True,
        log_every_iters=None,
    )
    tb_logger.close()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
@pytest.mark.skip(reason="Visdom is unmaintained and cannot be installed with modern packages")
def test_setup_visdom_logging(visdom_offline_logfile):
    vis_logger = _test_setup_logging(
        setup_logging_fn=setup_visdom_logging,
        kwargs_dict={"offline": True, "log_to_filename": visdom_offline_logfile},
        output_handler_cls=handlers.visdom_logger.OutputHandler,
        opt_params_handler_cls=handlers.visdom_logger.OptimizerParamsHandler,
        with_eval=False,
        with_optim=False,
    )
    vis_logger.close()

    vis_logger = _test_setup_logging(
        setup_logging_fn=setup_visdom_logging,
        kwargs_dict={"offline": True, "log_to_filename": visdom_offline_logfile},
        output_handler_cls=handlers.visdom_logger.OutputHandler,
        opt_params_handler_cls=handlers.visdom_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
    )
    vis_logger.close()


def test_setup_plx_logging():
    os.environ["POLYAXON_NO_OP"] = "1"

    _test_setup_logging(
        setup_logging_fn=setup_plx_logging,
        kwargs_dict={},
        output_handler_cls=handlers.polyaxon_logger.OutputHandler,
        opt_params_handler_cls=handlers.polyaxon_logger.OptimizerParamsHandler,
        with_eval=False,
        with_optim=False,
    )
    _test_setup_logging(
        setup_logging_fn=setup_plx_logging,
        kwargs_dict={},
        output_handler_cls=handlers.polyaxon_logger.OutputHandler,
        opt_params_handler_cls=handlers.polyaxon_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
    )


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_setup_mlflow_logging(dirname):
    mlf_logger = _test_setup_logging(
        setup_logging_fn=setup_mlflow_logging,
        kwargs_dict={"tracking_uri": str(dirname / "p1")},
        output_handler_cls=handlers.mlflow_logger.OutputHandler,
        opt_params_handler_cls=handlers.mlflow_logger.OptimizerParamsHandler,
        with_eval=False,
        with_optim=False,
    )
    mlf_logger.close()
    mlf_logger = _test_setup_logging(
        setup_logging_fn=setup_mlflow_logging,
        kwargs_dict={"tracking_uri": str(dirname / "p2")},
        output_handler_cls=handlers.mlflow_logger.OutputHandler,
        opt_params_handler_cls=handlers.mlflow_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
    )
    mlf_logger.close()


def test_setup_wandb_logging(dirname):
    with patch("ignite.handlers.logger_utils.WandBLogger") as _:
        setup_wandb_logging(MagicMock())


def test_setup_logging_migration_aliases():
    import ignite.contrib.engines.common as common
    import ignite.handlers.logger_utils as logger_mod

    assert common._setup_logging is logger_mod._setup_logging
    assert common.setup_tb_logging is logger_mod.setup_tb_logging
    assert common.setup_visdom_logging is logger_mod.setup_visdom_logging
    assert common.setup_mlflow_logging is logger_mod.setup_mlflow_logging
    assert common.setup_neptune_logging is logger_mod.setup_neptune_logging
    assert common.setup_wandb_logging is logger_mod.setup_wandb_logging
    assert common.setup_plx_logging is logger_mod.setup_plx_logging
    assert common.setup_clearml_logging is logger_mod.setup_clearml_logging
    assert common.setup_trains_logging is logger_mod.setup_trains_logging


def test_setup_clearml_logging():
    handlers.clearml_logger.ClearMLLogger.set_bypass_mode(True)

    with pytest.warns(UserWarning, match=r"running in bypass mode"):
        clearml_logger = _test_setup_logging(
            setup_logging_fn=setup_clearml_logging,
            kwargs_dict={},
            output_handler_cls=handlers.clearml_logger.OutputHandler,
            opt_params_handler_cls=handlers.clearml_logger.OptimizerParamsHandler,
            with_eval=False,
            with_optim=False,
        )
        clearml_logger.close()
        clearml_logger = _test_setup_logging(
            setup_logging_fn=setup_clearml_logging,
            kwargs_dict={},
            output_handler_cls=handlers.clearml_logger.OutputHandler,
            opt_params_handler_cls=handlers.clearml_logger.OptimizerParamsHandler,
            with_eval=True,
            with_optim=True,
        )
        clearml_logger.close()
        clearml_logger = _test_setup_logging(
            setup_logging_fn=setup_trains_logging,
            kwargs_dict={},
            output_handler_cls=handlers.clearml_logger.OutputHandler,
            opt_params_handler_cls=handlers.clearml_logger.OptimizerParamsHandler,
            with_eval=True,
            with_optim=True,
        )
        clearml_logger.close()

    with pytest.warns(UserWarning, match="setup_trains_logging was renamed to setup_clearml_logging"):
        clearml_logger = _test_setup_logging(
            setup_logging_fn=setup_trains_logging,
            kwargs_dict={},
            output_handler_cls=handlers.clearml_logger.OutputHandler,
            opt_params_handler_cls=handlers.clearml_logger.OptimizerParamsHandler,
            with_eval=True,
            with_optim=True,
        )
        clearml_logger.close()


def test_setup_neptune_logging(dirname):
    npt_logger = _test_setup_logging(
        setup_logging_fn=setup_neptune_logging,
        kwargs_dict={"mode": "offline"},
        output_handler_cls=handlers.neptune_logger.OutputHandler,
        opt_params_handler_cls=handlers.neptune_logger.OptimizerParamsHandler,
        with_eval=False,
        with_optim=False,
    )
    npt_logger.close()
    npt_logger = _test_setup_logging(
        setup_logging_fn=setup_neptune_logging,
        kwargs_dict={"mode": "offline"},
        output_handler_cls=handlers.neptune_logger.OutputHandler,
        opt_params_handler_cls=handlers.neptune_logger.OptimizerParamsHandler,
        with_eval=True,
        with_optim=True,
    )
    npt_logger.close()
