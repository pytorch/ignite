from __future__ import annotations

import collections
from collections import defaultdict, UserString
from collections.abc import Mapping, Sequence
from enum import Enum
import json
import os
from threading import RLock
import time
from typing import Any

from ignite.engine import Engine, Events

_DEFAULT_METRIC_PATH = '/tmp/hypertune/output.metrics'

_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE = 100

def _get_loss_from_output(output: Sequence[Mapping[str, Any]]) -> Any:
    return output[0]["loss"]

class MetricLoggerKeys(Enum):
    METRICS = "Metrics"
    LOSS = "Loss"


class HPOLogger(object):
    """
    Makes selected metric accessible for use by GCP Vertex AI hyperparameter tuning jobs. By adding only this
    single HPO-logger it unlocks hyperparam screening for Ignite

    Example::
        # construct an evaluator saving metric values
        val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
        evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
        evaluator.logger = setup_logger("evaluator")

        # construct the logger and associate with evaluator to extract metric values from
        hpo_logger = HPOLogger(evaluator = evaluator, metric_tag= 'nll')

        # construct the trainer with the logger passed in as a handler so that it logs loss values
        trainer.add_event_handler(Events.EPOCH_COMPLETED, hpo_logger)

        # run training
        trainer.run(train_loader, max_epochs=epochs)

    Args:
        evaluator: Evaluator to consume metric results from at the end of its evaluation run
        metric_tag: Converts the metric value coming from the trainer/evaluator's state into a storable value
    """

    def __init__(
        self,
        evaluator: Engine | None = None,
        metric_tag: UserString = 'training/hptuning/metric'
    ) -> None:
        self.loss: list = []
        self.metrics: defaultdict = defaultdict(list)
        self.iteration = 0
        self.lock = RLock()
        self.metric_tag = metric_tag

        if evaluator is not None:
            self.attach_evaluator(evaluator)

        """Constructor."""
        self.metric_path = os.environ.get('CLOUD_ML_HP_METRIC_FILE',
                                          _DEFAULT_METRIC_PATH)
        if not os.path.exists(os.path.dirname(self.metric_path)):
            os.makedirs(os.path.dirname(self.metric_path))

        self.trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', 0)
        self.metrics_queue = collections.deque(
            maxlen=_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE)
    
    def _dump_metrics_to_file(self):
        with open(self.metric_path, 'w') as metric_file:
            for metric in self.metrics_queue:
                metric_file.write(json.dumps(metric, sort_keys=True) + '\n')

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def attach_evaluator(self, evaluator: Engine) -> None:
        """
        Attach event  handlers to the given evaluator to log metric values from it.

        Args:
            evaluator: Ignite Engine implementing network evaluation
        """
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self.log_metrics)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        with self.lock:
            self.iteration = engine.state.iteration
            lossval = engine.state.output

            metric_value = float(lossval)

            metric_body = {
                'timestamp': time.time(),
                'trial': str(self.trial_id),
                self.metric_tag: str(metric_value),
                'global_step': str(int(self.iteration) if self.iteration else 0),
                'checkpoint_path': ''
            }
            self.metrics_queue.append(metric_body)
            self._dump_metrics_to_file()

    def log_metrics(self, engine: Engine) -> None:
        """
        Log metrics from the given Engine's state member.

        Args:
            engine: Ignite Engine to log from
        """
        with self.lock:
            for m, v in engine.state.metrics.items():
                self.metrics[m].append((self.iteration, v))

    def state_dict(self):
        return {MetricLoggerKeys.LOSS: self.loss, MetricLoggerKeys.METRICS: self.metrics}

    def load_state_dict(self, state_dict):
        self.loss[:] = state_dict[MetricLoggerKeys.LOSS]
        self.metrics.clear()
        self.metrics.update(state_dict[MetricLoggerKeys.METRICS])

hpologger = HPOLogger()