from __future__ import print_function

from functools import partial

from ignite.evaluator import Evaluator
from ignite.trainer import Trainer
from ignite.history import History


def _log_engine_history_average(engine, metric_name, msg_avg_type, history_avg_fn, logger):
    total_iterations = len(engine.dataloader)
    current_iteration = (engine.current_iteration - 1) % total_iterations + 1
    history_average = history_avg_fn(engine.history)
    msg_prefix = ""

    if isinstance(engine, Trainer):
        msg_prefix = "Training Epoch[{}/{}] ".format(engine.current_epoch, engine.max_epochs)
    elif isinstance(engine, Evaluator):
        msg_prefix = "Evaluation "

    log_str = "{}Iteration[{}/{} ({:.2f}%)]\t{} {}: {:.4f}" \
        .format(msg_prefix,
                current_iteration, total_iterations, (100. * current_iteration) / total_iterations,
                metric_name, msg_avg_type, history_average)
    logger(log_str)


def log_simple_moving_average(engine, window_size, history_transform=lambda x: x,
                              should_log=lambda engine: True, metric_name="", logger=print):
    if should_log(engine):
        _log_engine_history_average(engine, metric_name, "Simple Moving Average",
                                    partial(History.simple_moving_average, window_size=window_size,
                                            transform=history_transform),
                                    logger)


def log_weighted_moving_average(engine, window_size, weights, history_transform=lambda x: x,
                                should_log=lambda engine: True, metric_name="", logger=print):
    if should_log(engine):
        _log_engine_history_average(engine, metric_name, "Weighted Moving Average",
                                    partial(History.weighted_moving_average, window_size=window_size,
                                            weights=weights, transform=history_transform),
                                    logger)


def log_exponential_moving_average(engine, window_size, alpha, history_transform=lambda x: x,
                                   should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(engine):
        _log_engine_history_average(engine, metric_name, "Exponential Moving Average",
                                    partial(History.exponential_moving_average, window_size=window_size,
                                            alpha=alpha, transform=history_transform),
                                    logger)
