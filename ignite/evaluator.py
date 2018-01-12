import time

from torch.autograd import Variable

from ignite.history import History
from ignite.engine import Engine

from enum import Enum


class EvaluationEvents(Enum):
    EVALUATION_STARTING = "evaluation_starting"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_ITERATION_STARTED = "evaluation_iteration_started"
    EVALUATION_ITERATION_COMPLETED = "evaluation_iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


def _to_hours_mins_secs(time_taken):
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


class Evaluator(Engine):
    def __init__(self, inference_function):
        super(Evaluator, self).__init__(EvaluationEvents)

        self._inference_function = inference_function
        self.history = History()
        self.current_evaluation_iteration = 0
        self.should_terminate = False

    def run(self, dataset):
        """ Evaluates the dataset"""
        if self._inference_function is None:
            raise ValueError("Evaluator must have an inference function in order to evaluate")

        self.current_evaluation_iteration = 0
        self._fire_event(EvaluationEvents.EVALUATION_STARTING)
        start_time = time.time()

        for _, batch in enumerate(dataset, 1):
            self._fire_event(EvaluationEvents.EVALUATION_ITERATION_STARTED)
            step_result = self._inference_function(batch)
            if step_result is not None:
                self.history.append(step_result)

            self.current_evaluation_iteration += 1
            self._fire_event(EvaluationEvents.EVALUATION_ITERATION_COMPLETED)
            if self.should_terminate:
                break

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info("Evaluation Complete. Time taken: %02d:%02d:%02d", hours, mins, secs)

        self._fire_event(EvaluationEvents.EVALUATION_COMPLETED)


def create_supervised(model, cuda=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (torch.nn.Module): the model to train
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Trainer: a trainer instance with supervised inference function
    """
    def _prepare_batch(batch):
        x, y = batch
        if cuda:
            x, y = x.cuda(), y.cuda()
        return Variable(x, volatile=True), Variable(y, volatile=True)

    def _inference(batch):
        model.eval()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        return y_pred.data.cpu(), y.data.cpu()

    return Evaluator(_inference)

