from torch.autograd import Variable


class Supervised(object):
    """
    Provides training and validation update functions for standard supervised models.

    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)

    """
    def __init__(self, model, optimizer, loss_fn, cuda=False):
        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._cuda = cuda

    def update(self, batch):
        """
        Training update function.

        Args:
            batch (tuple): the model input and target

        Returns:
            float: the batch loss
        """
        self._model.train()
        self._optimizer.zero_grad()
        x, y = self._prepare_batch(batch)
        y_pred = self._model(x)
        loss = self._loss_fn(y_pred, y)
        loss.backward()
        self._optimizer.step()
        return loss.data[0]

    def predict(self, batch):
        """
        Validation update function.

        Args:
            batch (tuple): the model input and target

        Returns:
            tuple: the prediction and the target
        """
        self._model.eval()
        x, y = self._prepare_batch(batch)
        y_pred = self._model(x)
        return y_pred.data.cpu(), y.data.cpu()

    def _prepare_batch(self, batch):
        x, y = batch
        if self._cuda:
            x, y = x.cuda(), y.cuda()
        return Variable(x), Variable(y)
