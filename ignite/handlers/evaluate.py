class Evaluate(object):
    """
    Triggers validation and optionally clears history.

    Args:
        evaluator (Evaluator): the evaluator to run
        evaluation_data (iterable): the data to validate on
        iteration_interval (int, optional): number of iterations between validations
        epoch_interval (int, optional): number of epochs between validations
        clear_history (bool, optional): whether or not to clear history before each
            validation (default: True)
    """
    def __init__(self, evaluator, evaluation_data, iteration_interval=None, epoch_interval=None,
                 clear_history=True):
        self._evaluator = evaluator
        self._evaluation_data = evaluation_data
        self._iteration_interval = iteration_interval
        self._epoch_interval = epoch_interval
        self._clear_history = clear_history

        if iteration_interval and epoch_interval:
            raise ValueError('you must pass only one of (iteration_interval, epoch_interval)')
        if (iteration_interval is None) and (epoch_interval is None):
            raise ValueError('you must pass one of (iteration_interval, epoch_interval)')

    def __call__(self, trainer):
        if self._should_evaluate(trainer):
            if self._clear_history:
                self._evaluator.history.clear()
            self._evaluator.run(self._validation_data)

    def _should_evaluate(self, trainer):
        if self._iteration_interval:
            return (trainer.current_iteration % self._iteration_interval) == 0
        if self._epoch_interval:
            return(trainer.current_epoch % self._epoch_interval) == 0
        return False
