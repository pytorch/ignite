try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("This contrib module requires tensorboardX to be installed.")

from ignite.engine import Engine, Events
from torch import nn as nn
import os


class TensorboardLogger(object):
    """
    TensorBoard handler to log metrics related to training, validation and model.

    Examples:

        Create a TensorBoard summary that visualizes metrics related to training,
        validation, and model parameters, by simply attaching the handler to your engine.

        ..code-block:: python

            tbLogger = TensorboardLogger()

            tbLogger.attach(engine=trainer,
                            mode='epoch',
                            model=model,
                            dataloader=data,
                            use_metrics=True,
                            state_keys=['reward'],
                            histogram_freq=1,
                            write_grads=True)

            tbLogger.attach_evaluator(engine=evaluator,
                                      use_metrics=True,
                                      state_keys=['reward'])

    Note:
        When adding tensorboard event handler to an engine, it is recommended that the evaluator
        is run before the code block above.

        Depending on the engines this callback is attached to, SummaryWriter's are saved in self.writer
        which is a dictionary of keys 'train' or 'eval. If custom functions are needed, use
        tbLogger.writer['train'].add_*.
    """

    def __init__(self, log_dir=None):
        self.attached = False
        if not log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
        self.log_dir = log_dir
        self.writer = {}

    def _create_writer(self, mode):
        self.writer[mode] = SummaryWriter(log_dir=os.path.join(self.log_dir, mode))

    def _close(self, engine):
        """
        Closes Summary Writer
        """
        for key, writer in self.writer.items():
            writer.close()

    def _write_graph(self, engine, model, dataloader):
        """
        Create computational graph of model.

        Args:
            model (nn.Module): model to be written
            dataloader (torch.utils.DataLoader): data loader for training data
        """
        writer = self.writer['train']
        x, y = next(iter(dataloader))
        x = x.cuda() if next(model.parameters()).is_cuda else x
        writer.add_graph(model, x)
        del x, y

    def _plot_metrics(self, engine, writer, mode='epoch'):
        """
        Plots engine.state.metrics on epoch or iteration.

        Args:
            engine (ignite.Engine): training engine or evaluator engine
            mode (str): 'epoch' or 'iteration'
        """

        global_step = self.engine.state.epoch if mode == 'epoch' else self.engine.state.iteration

        for key, value in engine.state.metrics.items():
            writer.add_scalar(tag=key,
                              scalar_value=value,
                              global_step=global_step)

    def _plot_state_keys(self, engine, writer, state_keys, mode='epoch'):
        """
        Plots engine.state attributes on epoch or iteration.

        Args:
            engine (ignite.Engine): training engine or evaluator engine
            state_keys (list): list of keys of engine.state attributes
            mode (str): 'epoch' or 'iteration'
        """

        global_step = self.engine.state.epoch if mode == 'epoch' else self.engine.state.iteration
        for key in state_keys:

            if hasattr(engine.state, key):
                state = getattr(engine.state, key)
            else:
                raise KeyError('{} not an attribute of engine.state'.format(key))

            if isinstance(state, (int, float)):
                writer.add_scalar(tag=key,
                                  scalar_value=state,
                                  global_step=global_step)
            else:
                raise ValueError('engine.state.{} should be of instance int or float.'.format(key))

    def _plot_output(self, engine, writer, output_transform):
        global_step = self.engine.state.iteration

        output = output_transform(engine.state.output)

        if not isinstance(output, dict):
            raise ValueError("output_transform must transform engine.state.output into a dictionary.")

        for key, value in output.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(tag=key,
                                  scalar_value=value,
                                  global_step=global_step)
            else:
                raise ValueError('engine.state.output[\'{}\'] should be of instance int or float.'.format(key))

    def _update_iteration(self, engine, mode, use_metrics, use_output, output_transform, state_keys, learning):

        writer = self.writer[learning]

        if mode == 'iteration':
            if use_metrics:
                self._plot_metrics(engine, writer, mode='iteration')
            if state_keys:
                self._plot_state_keys(engine, writer, state_keys, mode='iteration')

        if use_output:
            self._plot_output(engine, writer, output_transform)

    def _update_epoch(self, engine, mode, use_metrics, state_keys, model, histogram_freq, write_grads, learning):

        writer = self.writer[learning]

        if mode == 'epoch':
            if use_metrics:
                self._plot_metrics(engine, writer, mode='epoch')

            if state_keys:
                self._plot_state_keys(engine, writer, state_keys, mode='epoch')

        if histogram_freq > 0 and self.engine.state.epoch % histogram_freq == 0:
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                if param.requires_grad:
                    writer.add_histogram(tag=name,
                                         values=param.cpu().detach().numpy().flatten(),
                                         global_step=self.engine.state.epoch)
                    if write_grads:
                        writer.add_histogram(tag=name + '_grad',
                                             values=param.grad.cpu().detach().numpy().flatten(),
                                             global_step=self.engine.state.epoch)

    def attach(self,
               engine,
               mode='iteration',
               model=None,
               dataloader=None,
               write_graph=False,
               use_output=True,
               output_transform=lambda x: x,
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
            dataloader (generator): (Optional) generator used to train model.
            write_graph (bool): (Optional) True, if model graph should be written.
            use_metrics (bool): (Optional) True, if metrics for engine and evaluator should be visualized.
            use_output (bool): (Optional) True, if engine.state.output should be plotted.
            output_transform (function): (Optional) Callable function to transform engine.state.output
                                         into a dictionary of integers and floats.
            state_keys (list): (Optional) list of string, state attributes to be visualized.
            histogram_freq (int): (Optional) frequency histograms should be plotted.
            write_grads (bool): (Optional) True, if gradients to model parameters should be visualized.
        """

        self.attached = True
        self.engine = engine
        self._create_writer(mode='train')

        if mode not in ['iteration', 'epoch']:
            raise ValueError("mode should be iteration or epoch, got {} instead.".format(mode))

        if not isinstance(model, nn.Module):
            raise TypeError("model should be of type nn.Module, got {} instead.".format(type(model)))

        try:
            loader = iter(dataloader)
        except:
            raise TypeError("dataloader should be an iterable.")

        if not isinstance(write_graph, bool):
            raise TypeError("write_graph should be a boolean, got {} instead.".format(type(write_graph)))

        if not isinstance(use_output, bool):
            raise TypeError("use_output should be a boolean, got {} instead.".format(type(use_output)))

        if not callable(output_transform):
            raise ValueError("output_transform should be callable, got {} instead.".format(type(output_transform)))

        if not isinstance(use_metrics, bool):
            raise TypeError("use_metrics should be a boolean, got {} instead.".format(type(use_metrics)))

        if state_keys and not isinstance(state_keys, list):
            raise TypeError("state_keys should be a list, got {} instead.".format(type(state_keys)))

        if not isinstance(histogram_freq, int):
            raise TypeError("histogram_freq should be an int, got {} instead.".format(type(histogram_freq)))

        if not isinstance(write_grads, bool):
            raise TypeError("write_grads should be a boolean, got {} instead.".format(type(write_grads)))

        if write_graph:
            engine.add_event_handler(Events.STARTED,
                                     self._write_graph,
                                     model,
                                     dataloader)

        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._update_iteration,
                                 mode,
                                 use_metrics,
                                 use_output,
                                 output_transform,
                                 state_keys,
                                 learning='train')

        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._update_epoch,
                                 mode,
                                 use_metrics,
                                 state_keys,
                                 model,
                                 histogram_freq,
                                 write_grads,
                                 learning='train')
        engine.add_event_handler(Events.COMPLETED, self._close)

    def attach_evaluator(self,
                         engine,
                         mode='eval',
                         use_metrics=False,
                         state_keys=None):
        """
        Attaches the TensorBoard event handler to an evaluator engine objects.

        Only used to plot metrics and state_keys.

        Uses training engine.state.epoch or engine.state.iteration as global_step.

        Can be attached to multiple evaluators. For example: evaluator run on training dataset and validation datasets.

        Args:
            engine (Engine): engine object.
            mode (str): (Optional) text to describe evaluator.
            use_metrics (bool): (Optional) True, if metrics for engine and evaluator should be visualized.
            state_keys (list): (Optional) list of string, state attributes to be visualized.
        """
        if not self.attached:
            raise ValueError("Please attach handler to training engine before attaching to evaluator engine.")

        self._create_writer(mode=mode)

        if not isinstance(use_metrics, bool):
            raise TypeError("use_metrics should be a boolean, got {} instead.".format(type(use_metrics)))

        if state_keys and not isinstance(state_keys, list):
            raise TypeError("state_keys should be a list, got {} instead.".format(type(state_keys)))

        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._update_epoch,
                                 mode='epoch',
                                 use_metrics=use_metrics,
                                 state_keys=state_keys,
                                 model=None,
                                 histogram_freq=0,
                                 write_grads=False,
                                 learning=mode)
