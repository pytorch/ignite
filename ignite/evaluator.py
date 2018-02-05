import time

from torch.autograd import Variable
from ignite.engine import Engine, Events
from ignite._utils import _to_hours_mins_secs, to_variable


__all__ = ["Evaluator", "create_supervised_evaluator"]


class Evaluator(Engine):
    def __init__(self, inference_function):
        super(Evaluator, self).__init__(inference_function)

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        if event_name == Events.EPOCH_STARTED or event_name == Events.EPOCH_COMPLETED:
            raise ValueError("Evaluator does not fire event {} ".format(event_name))

        super(Evaluator, self).add_event_handler(event_name, handler, *args, **kwargs)

    def run(self, dataset):
        self.dataset = dataset
        self.current_iteration = 0

        self._fire_event(Events.STARTED)
        start_time = time.time()
        try:
            self._run_once_on_dataset(dataset)
        except BaseException as e:
            self._logger.error("Evaluation is terminating due to exception: %s", str(e))
            self._fire_event(Events.EXCEPTION_RAISED)
            raise e
        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self._logger.info("Evaluation Complete. Time taken: %02d:%02d:%02d", hours, mins, secs)

        self._fire_event(Events.COMPLETED)


def create_supervised_evaluator(model, cuda=False):
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
        x = to_variable(x, cuda=cuda, volatile=True)
        y = to_variable(y, cuda=cuda, volatile=True)
        return x, y

    def _inference(batch):
        model.eval()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        return y_pred.data.cpu(), y.data.cpu()

    return Evaluator(_inference)
