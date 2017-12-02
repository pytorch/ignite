def log_training_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                       should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        iterations_per_epoch = len(trainer.training_data)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%)]\t{}Simple Moving Average: {:.4f}" \
            .format(trainer.current_epoch, trainer.max_epochs, trainer.current_iteration,
                    iterations_per_epoch, (100 * trainer.current_iteration) / float(iterations_per_epoch),
                    metric_name + " ",
                    trainer.training_history.simple_moving_average(window_size, history_transform))
        logger(log_str)


def log_validation_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = len(trainer.validation_data)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Simple Moving Average: {:.4f}" \
            .format(trainer.current_validation_iteration, total_iterations,
                    (100 * trainer.current_validation_iteration) / float(total_iterations),
                    metric_name + " ",
                    trainer.validation_history.simple_moving_average(window_size, history_transform))
        logger(log_str)


def log_training_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        iterations_per_epoch = len(trainer.training_data)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%}]\t{}Weighted Moving Average: {:.4f}" \
            .format(trainer.current_epoch, trainer.max_epochs, trainer.current_iteration,
                    iterations_per_epoch, (100 * trainer.current_iteration) / float(iterations_per_epoch),
                    metric_name + " ",
                    trainer.training_history.weighted_moving_average(window_size, weights, history_transform))
        logger(log_str)


def log_validation_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                           should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = len(trainer.validation_data)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Weighted Moving Average: {:.4f}" \
            .format(trainer.current_validation_iteration, total_iterations,
                    (100 * trainer.current_iteratiocurrent_validation_iteration) / float(total_iterations),
                    metric_name + " ",
                    trainer.validation_history.weighted_moving_average(window_size, weights, history_transform))
        logger(log_str)


def log_training_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                            should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        iterations_per_epoch = len(trainer.training_data)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%)]\t{}Exponential Moving Average: {:.4f}" \
            .format(trainer.current_epoch, trainer.max_epochs, trainer.current_iteration,
                    iterations_per_epoch, (100 * trainer.current_iteration) / float(iterations_per_epoch),
                    metric_name + " ",
                    trainer.training_history.exponential_moving_average(window_size, alpha, history_transform))
        logger(log_str)


def log_validation_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                              should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = len(trainer.validation_data)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Exponential Moving Average: {:.4f}" \
            .format(trainer.current_validation_iteration, total_iterations,
                    (100 * trainer.current_iteratiocurrent_validation_iteration) / float(total_iterations),
                    metric_name + " ",
                    trainer.validation_history.exponential_moving_average(window_size, alpha, history_transform))
        logger(log_str)
