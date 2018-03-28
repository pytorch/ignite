from ignite.engines.engine import Engine, Events, State
from ignite._utils import to_variable, to_tensor


def _prepare_batch(batch, cuda, volatile=False):
    x, y = batch
    x = to_variable(x, cuda=cuda, volatile=volatile)
    y = to_variable(y, cuda=cuda, volatile=volatile)
    return x, y


def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Engine: a trainer engine with supervised update function
    """
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, cuda)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data.cpu()[0]

    return Engine(_update)


def create_supervised_evaluator(model, metrics={}, cuda=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _inference(engine, batch):
        model.eval()
        x, y = _prepare_batch(batch, cuda, volatile=True)
        y_pred = model(x)
        return to_tensor(y_pred, cpu=not cuda), to_tensor(y, cpu=not cuda)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
