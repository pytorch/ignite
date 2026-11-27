"""GCP Vertex AI logger and its helper handlers."""

import json
import os
import time
import warnings
from collections import deque
from collections.abc import Callable
from typing import Any

from torch.optim import Optimizer

from ignite.engine import Engine, Events
from ignite.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler

__all__ = ["VertexAILogger", "OutputHandler", "OptimizerParamsHandler"]

_DEFAULT_HYPERPARAMETER_METRIC_TAG = "training/hptuning/metric"
_DEFAULT_METRIC_PATH = "/tmp/hypertune/output.metrics"
_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE = 100


class VertexAILogger(BaseLogger):
    """GCP Vertex AI handler to report metrics for Hyperparameter Tuning jobs.

    This logger writes metrics in the format expected by Vertex AI's hyperparameter
    tuning service. It does not require any additional dependencies and works by
    writing newline-delimited JSON to the metric file path specified by the
    ``CLOUD_ML_HP_METRIC_FILE`` environment variable (default:
    ``/tmp/hypertune/output.metrics``).

    Vertex AI reads metrics from this file to evaluate trial performance during
    hyperparameter tuning jobs. This enables Ignite-based training code to be
    used with Vertex AI's hyperparameter tuning orchestration.

    Examples:
        .. code-block:: python

            from ignite.handlers.gcp_vertex_logger import VertexAILogger

            # Create a logger
            vertex_logger = VertexAILogger()

            # Attach the logger to the trainer to report accuracy after each epoch
            vertex_logger.attach_output_handler(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["accuracy"],
                global_step_transform=lambda engine, event_name: engine.state.epoch,
            )

            # Attach the logger to report metrics at the end of training
            vertex_logger.attach_output_handler(
                trainer,
                event_name=Events.COMPLETED,
                tag="training",
                metric_names=["accuracy"],
                global_step_transform=lambda engine, event_name: engine.state.epoch,
            )

    .. versionadded:: 0.5.0
    """

    def __init__(self, **kwargs: Any):
        self.metric_path = os.environ.get(
            "CLOUD_ML_HP_METRIC_FILE", _DEFAULT_METRIC_PATH
        )
        os.makedirs(os.path.dirname(self.metric_path), exist_ok=True)
        self.trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", 0)
        self.metrics_queue = deque(maxlen=_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE)
        self._kwargs = kwargs

    def _dump_metrics_to_file(self) -> None:
        with open(self.metric_path, "w") as metric_file:
            for metric in self.metrics_queue:
                metric_file.write(json.dumps(metric, sort_keys=True) + "\n")

    def report_hyperparameter_tuning_metric(
        self,
        hyperparameter_metric_tag: str,
        metric_value: float,
        global_step: int | None = None,
        checkpoint_path: str = "",
    ) -> None:
        """Report a hyperparameter tuning metric to Vertex AI.

        This method can be used directly if you need custom control over metric reporting.

        Args:
            hyperparameter_metric_tag: The name of the metric to report (e.g., "accuracy").
            metric_value: The value of the metric.
            global_step: The current training step.
            checkpoint_path: Optional path to a checkpoint file.
        """
        metric_value = float(metric_value)
        metric_tag = (
            hyperparameter_metric_tag
            if hyperparameter_metric_tag
            else _DEFAULT_HYPERPARAMETER_METRIC_TAG
        )

        metric_body = {
            "timestamp": time.time(),
            "trial": str(self.trial_id),
            metric_tag: str(metric_value),
            "global_step": str(int(global_step) if global_step else 0),
            "checkpoint_path": checkpoint_path,
        }
        self.metrics_queue.append(metric_body)
        self._dump_metrics_to_file()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(
        self, *args: Any, **kwargs: Any
    ) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics to Vertex AI.

    Args:
        tag: common title for all produced plots. For example, "training".
        metric_names: list of metric names to log or a string "all" to log all available
            metrics.
        output_transform: output transform function to prepare `engine.state.output` as a number.
            For example, ``output_transform = lambda output: output``.
            This function can also return a dictionary, e.g ``{"loss": loss1, "another_loss": loss2}``
            to label the plot with corresponding keys.
        global_step_transform: global step transform function to output a desired global step.
            Input of the function is ``(engine, event_name)``. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.handlers.gcp_vertex_logger.global_step_from_engine`.
        state_attributes: list of state attributes to log.

    Examples:
        .. code-block:: python

            from ignite.handlers.gcp_vertex_logger import VertexAILogger, OutputHandler

            vertex_logger = VertexAILogger()

            vertex_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["accuracy", "nll"],
                    global_step_transform=lambda *_: trainer.state.iteration,
                ),
                event_name=Events.EPOCH_COMPLETED,
            )
    """

    def __init__(
        self,
        tag: str,
        metric_names: str | list[str] | None = None,
        output_transform: Callable | None = None,
        global_step_transform: Callable[[Engine, str | Events], int] | None = None,
        state_attributes: list[str] | None = None,
    ):
        super().__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )

    def __call__(self, engine: Engine, logger: VertexAILogger, event_name: str | Events) -> None:
        if not isinstance(logger, VertexAILogger):
            raise RuntimeError(
                f"Handler '{self.__class__.__name__}' works only with VertexAILogger."
            )

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}. "
                "Please check the output of global_step_transform."
            )

        metrics = self._setup_output_metrics_state_attrs(
            engine, log_text=True, key_tuple=False
        )
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=metric_name,
                    metric_value=metric_value,
                    global_step=global_step,
                )


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters to Vertex AI.

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator".
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str = "lr",
        tag: str | None = None,
    ):
        super().__init__(optimizer, param_name, tag)

    def __call__(
        self, engine: Engine, logger: VertexAILogger, event_name: str | Events
    ) -> None:
        if not isinstance(logger, VertexAILogger):
            raise RuntimeError(
                "Handler OptimizerParamsHandler works only with VertexAILogger"
            )

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(
                param_group[self.param_name]
            )
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        for param_name, param_value in params.items():
            logger.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=param_name,
                metric_value=param_value,
                global_step=global_step,
            )
