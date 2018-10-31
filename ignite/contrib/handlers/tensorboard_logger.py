try:
    from tensorboardX import SummaryWriter
except:
    raise RuntimeError("This contrib module requires tensorboardX to be installed.")

from ignite.engine import Engine, Events
from torch import nn as nn


class TensorboardLogger(object):
    """
    TensorBoard handler to log metrics related to training, validation and model.

    Examples:

        Create a TensorBoard summary that visualizes metrics related to training,
        validation, and model parameters, by simply attaching the handler to your engine.

        ..code-block:: python

            tbLogger = TensorboardLogger()
            tbLogger.attach(engine=trainer,
                            mode='iteration',
                            model=model,
                            input_shape=[1, 28, 28],
                            write_graph=True,
                            validation_evaluator=evaluator,
                            use_metrics=True,
                            histogram_freq=1,
                            write_grads=True)
    Note:
        When adding tensorboard event handler to an engine, it is recommended that the evaluator
        is run before the code block above. If custom functions are needed, use tbLogger.writer.add_*.
    """

    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(log_dir=log_dir)

    def _close(self, engine):
        """
        Closes Summary Writer
        """
        self.writer.close()

    def write_graph(self, model, dataloader):
        """
        Plots engine.state.metrics on epoch or iteration.

        Args:
            model (nn.Module): model to be written
            dataloader (torch.utils.DataLoader): data loader for training data
        """

        x, y = next(iter(dataloader))
        x = x.cuda() if next(model.parameters()).is_cuda else x
        self.writer.add_graph(model, x)

    def plot_metrics(self, engine, name, mode='epoch'):
        """
        Plots engine.state.metrics on epoch or iteration.

        Args:
            engine (ignite.Engine): training engine or evaluator engine
            name (str): name of trainer or evaluator
            mode (str): 'epoch' or 'iteration'
        """

        global_step = self.engine.state.epoch if mode == 'epoch' else self.engine.state.iteration

        for key, value in engine.state.metrics.items():
            self.writer.add_scalar(tag='/'.join([name, key]),
                                   scalar_value=value,
                                   global_step=global_step)

    def plot_state_keys(self, engine, state_keys, name, mode='epoch'):
        """
        Plots engine.state attributes on epoch or iteration.

        Args:
            engine (ignite.Engine): training engine or evaluator engine
            state_keys (list): list of keys of engine.state attributes
            name (str): name of trainer or evaluator
            mode (str): 'epoch' or 'iteration'
        """

        global_step = self.engine.state.epoch if mode == 'epoch' else self.engine.state.iteration
        for key in state_keys:

            if hasattr(engine.state, key):
                state = getattr(engine.state, key)
            else:
                raise KeyError('{} not an attribute of engine.state'.format(key))

            if isinstance(state, (int, float)):
                self.writer.add_scalar(tag='/'.join([name, key]),
                                       scalar_value=state,
                                       global_step=global_step)
            else:
                raise ValueError('engine.state.{} should be of instance int or float.'.format(key))

    def _update_iteration(self, engine, mode, use_metrics, state_keys):
        if use_metrics and mode == 'iteration':
            self.plot_metrics(engine, name='trainer', mode='iteration')
        if state_keys:
            self.plot_state_keys(engine, state_keys, name='trainer', mode='iteration')

    def _update_epoch(self, engine, mode, use_metrics, state_keys, model, histogram_freq, write_grads):
        if use_metrics and mode == 'epoch':
            self.plot_metrics(engine, name='trainer', mode='epoch')

        if state_keys:
            self.plot_state_keys(engine, state_keys, name='trainer', mode='epoch')

        if engine.state.epoch % histogram_freq == 0:
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                if param.requires_grad:
                    self.writer.add_histogram(tag=name,
                                              values=param.cpu().data.numpy().flatten(),
                                              global_step=engine.state.epoch)
                    if write_grads:
                        self.writer.add_histogram(tag=name + '_grad',
                                                  values=param.grad.cpu().data.numpy().flatten(),
                                                  global_step=engine.state.epoch)

    def attach(self,
               engine,
               mode='iteration',
               model=None,
               use_metrics=False,
               state_keys=None,
               histogram_freq=0,
               write_grads=False
               ):
        """
        Attaches the TensorBoard event handler to an engine object.

        Args:
            engine (Engine): engine object.
            mode (str): (Optional) iteration or epoch.
            model (nn.Module): (Optional) model to train.
            use_metrics (bool): (Optional) True, if metrics for engine and evaluator should be visualized.
            state_keys (list): (Optional) list of string, state attributes to be visualized.
            histogram_freq (int): (Optional) frequency histograms should be plotted.
            write_grads (bool): (Optional) True, if gradients to model parameters should be visualized.
        """

        self.engine = engine

        if mode not in ['iteration', 'epoch']:
            raise ValueError("mode should be iteration or epoch, got {} instead.".format(mode))

        if not isinstance(model, nn.Module):
            raise TypeError("model should be of type nn.Module, got {} instead.".format(type(model)))

        if not isinstance(use_metrics, bool):
            raise TypeError("write_graph should be a boolean, got {} instead.".format(type(use_metrics)))

        if state_keys and not isinstance(state_keys, list):
            raise TypeError("state_keys should be a list, got {} instead.".format(type(state_keys)))

        if not isinstance(histogram_freq, int):
            raise TypeError("histogram_freq should be an int, got {} instead.".format(type(histogram_freq)))

        if not isinstance(write_grads, bool):
            raise TypeError("write_grads should be a boolean, got {} instead.".format(type(write_grads)))

        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._update_iteration,
                                 mode,
                                 use_metrics,
                                 state_keys)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._update_epoch,
                                 mode,
                                 use_metrics,
                                 state_keys,
                                 model,
                                 histogram_freq,
                                 write_grads)
        engine.add_event_handler(Events.COMPLETED, self._close)
