from __future__ import print_function


def log_training_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                       should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):        
        current_iteration = trainer.current_iteration % (trainer.iterations_per_epoch + 1)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%)]\t{}Simple Moving Average: {:.4f}" \
            .format(trainer.current_epoch + 1, trainer.max_epochs, current_iteration,
                    trainer.iterations_per_epoch, (100. * current_iteration) / trainer.iterations_per_epoch,
                    metric_name + " ",
                    trainer.training_history.simple_moving_average(window_size, history_transform))
        logger(log_str)


def log_validation_simple_moving_average(trainer, window_size, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = trainer.total_validation_iterations
        current_iteration = trainer.current_iteration % (total_iterations + 1)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Simple Moving Average: {:.4f}" \
            .format(current_iteration + 1, total_iterations,
                    (100. * current_iteration) / total_iterations,
                    metric_name + " ",
                    trainer.validation_history.simple_moving_average(window_size, history_transform))
        logger(log_str)


def log_training_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                         should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):        
        current_iteration = trainer.current_iteration % (trainer.iterations_per_epoch + 1)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%}]\t{}Weighted Moving Average: {:.4f}" \
            .format(trainer.current_epoch + 1, trainer.max_epochs, current_iteration,
                    trainer.iterations_per_epoch, (100. * current_iteration) / trainer.iterations_per_epoch,
                    metric_name + " ",
                    trainer.training_history.weighted_moving_average(window_size, weights, history_transform))
        logger(log_str)


def log_validation_weighted_moving_average(trainer, window_size, weights, history_transform=lambda x: x,
                                           should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = trainer.total_validation_iterations
        current_iteration = trainer.current_iteration % (total_iterations + 1)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Weighted Moving Average: {:.4f}" \
            .format(current_iteration + 1, total_iterations,
                    (100. * current_iteration) / total_iterations,
                    metric_name + " ",
                    trainer.validation_history.weighted_moving_average(window_size, weights, history_transform))
        logger(log_str)


def log_training_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                            should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):        
        current_iteration = trainer.current_iteration % (trainer.iterations_per_epoch + 1)
        log_str = "Training Epoch[{}/{}] Iteration[{}/{} ({:.2f}%)]\t{}Exponential Moving Average: {:.4f}" \
            .format(trainer.current_epoch + 1, trainer.max_epochs, current_iteration,
                    trainer.iterations_per_epoch, (100. * current_iteration) / trainer.iterations_per_epoch,
                    metric_name + " ",
                    trainer.training_history.exponential_moving_average(window_size, alpha, history_transform))
        logger(log_str)


def log_validation_exponential_moving_average(trainer, window_size, alpha, history_transform=lambda x: x,
                                              should_log=lambda trainer: True, metric_name="", logger=print):
    if should_log(trainer):
        total_iterations = trainer.total_validation_iterations
        current_iteration = trainer.current_iteration % (total_iterations + 1)
        log_str = "Validation Iteration[{}/{} ({:.2f}%)]\t{}Exponential Moving Average: {:.4f}" \
            .format(trainer.current_validation_iteration + 1, total_iterations,
                    (100. * current_iteration) / total_iterations,
                    metric_name + " ",
                    trainer.validation_history.exponential_moving_average(window_size, alpha, history_transform))
        logger(log_str)
