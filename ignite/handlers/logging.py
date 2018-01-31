from __future__ import absolute_import, print_function

from functools import partial

from ..trainer.history import History


def _log_history_average(current_iteration, total_iterations,
                         msg_prefix, metric_name, msg_average_type, history_average,
                         logger):
    log_str = "{} Iteration[{}/{} ({:.2f}%)]\t{} {}: {:.4f}" \
        .format(msg_prefix,
                current_iteration, total_iterations, (100. * current_iteration) / total_iterations,
                metric_name, msg_average_type, history_average)
    logger(log_str)


def _log_training_history_average(trainer, metric_name, msg_average_type, history_average_fn, logger):
    current_epoch = trainer.current_epoch
    max_epochs = trainer.max_epochs
    total_iterations = trainer.iterations_per_epoch
    current_iteration = (trainer.current_iteration - 1) % total_iterations + 1
    history_average = history_average_fn(trainer.training_history)
    msg_prefix = "Training Epoch[{}/{}]".format(current_epoch + 1, max_epochs)
    _log_history_average(current_iteration, total_iterations,
                         msg_prefix, metric_name, msg_average_type, history_average, logger)


def _log_validation_history_average(trainer, metric_name, msg_average_type, history_average_fn, logger):
    total_iterations = trainer.total_validation_iterations
    current_iteration = (trainer.current_validation_iteration - 1) % total_iterations + 1
    history_average = history_average_fn(trainer.validation_history)
    msg_prefix = "Validation"
    _log_history_average(current_iteration, total_iterations,
                         msg_prefix, metric_name, msg_average_type, history_average, logger)


def log_training_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                       should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_training_history_average(trainer, metric_name,
                                      "Simple Moving Average",
                                      partial(History.simple_moving_average,
                                              window_size=window_size, transform=history_transform),
                                      logger)


def log_validation_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_validation_history_average(trainer, metric_name,
                                        "Simple Moving Average",
                                        partial(History.simple_moving_average,
                                                window_size=window_size, transform=history_transform),
                                        logger)


def log_training_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_training_history_average(trainer, metric_name,
                                      "Weighted Moving Average",
                                      partial(History.weighted_moving_average,
                                              window_size=window_size, weights=weights, transform=history_transform),
                                      logger)


def log_validation_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                           should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_validation_history_average(trainer, metric_name,
                                        "Weighted Moving Average",
                                        partial(History.weighted_moving_average,
                                                window_size=window_size, weights=weights, transform=history_transform),
                                        logger)


def log_training_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                            should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_training_history_average(trainer, metric_name,
                                      "Exponential Moving Average",
                                      partial(History.exponential_moving_average,
                                              window_size=window_size, alpha=alpha, transform=history_transform),
                                      logger)


def log_validation_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                              should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        _log_validation_history_average(trainer, metric_name,
                                        "Exponential Moving Average",
                                        partial(History.exponential_moving_average,
                                                window_size=window_size, alpha=alpha, transform=history_transform),
                                        logger)
