from ignite._utils import to_variable
from ignite.engine import Engine, Events, State

__all__ = ["Evaluator", "create_supervised_evaluator"]


class Evaluator(Engine):
    def add_event_handler(self, event_name, handler, *args, **kwargs):
        if event_name == Events.EPOCH_STARTED or event_name == Events.EPOCH_COMPLETED:
            raise ValueError("Evaluator does not fire event {} ".format(event_name))

        super(Evaluator, self).add_event_handler(event_name, handler, *args, **kwargs)

    def run(self, data):
        state = State(dataloader=data, metrics=[])
        self._fire_event(Events.STARTED, state)
        hours, mins, secs = self._run_once_on_dataset(state)
        self._logger.info("Evaluation Complete. Time taken: %02d:%02d:%02d", hours, mins, secs)
        self._fire_event(Events.COMPLETED, state)
        return state


def create_supervised_evaluator(model, metrics=[], cuda=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (torch.nn.Module): the model to train
        metrics (list of Metrics): a list of metrics to compute
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Evaluator: a evaluator instance with supervised inference function
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

    evaluator = Evaluator(_inference)

    for metric in metrics:
        metric.attach(evaluator)

    return evaluator
