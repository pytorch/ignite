from types import MethodType
import torch
from torchvision import utils
import numpy as np
from ignite.engine import Engine, Events
from tensorboardX import SummaryWriter


class TensorBoardX(object):
    """
        Event Handler to create TensorBoard Summary

        Args:
            log_dir (`str`): directory to save TensorBoard runs
            model (`torch.nn.Module`): the model to train
            input_shape (`list`): shape of input to model
            use_output (`bool`): True, if engine.state.output should be recorded (at the end of iteration)
            use_metrics (`bool`): True, if engine.state.metrics should be recorded (at the end of epoch)
            state_keys ('list'): list of strings that are attributes to engine.state.
                                Ensure that attribute is of `int` or `float` type.
            train_evaluator (`ignite.engine`): ignite engine that has been run on the training set.
            validation_evaluator (`ignite.engine`): ignite engine that has been run on the validation set.
            write_graph (`bool`): True, if graph should be recorded
            histogram_freq (`int`): frequency of epoch to record histogram of weights and gradients (if applicable)
            write_grads (`bool`): True, if gradients to model weights should be recorded
            write_images (`bool`): True, if model weights should be written as images

        Returns:
            SummaryWriter: a TensorBoard summary report
        """

    def __init__(self, 
                 log_dir=None,
                 model=None,
                 input_shape=None,
                 use_output=False,
                 use_metrics=False,
                 state_keys=None,
                 train_evaluator=None,
                 validation_evaluator=None,
                 write_graph=False,
                 histogram_freq=0,
                 write_grads=False,
                 write_images=False):

        self.log_dir = log_dir
        self.model = model
        self.input_shape = input_shape
        self.use_output = use_output
        self.use_metrics = use_metrics
        self.state_keys = state_keys

        self.train_evaluator = train_evaluator
        self.validation_evaluator = validation_evaluator

        self.write_graph = write_graph
        self.histogram_freq = histogram_freq
        self.write_grads = write_grads
        self.write_images = write_images

    def _on_start(self, engine):
        """
        This function creates the directory for the run and creates a TensorBoard graph using the model and input_shape.
        """
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.write_graph:
            x_shape = [1] + self.input_shape
            x = torch.zeros(*x_shape)
            self.is_cuda = False
            if next(self.model.parameters()).is_cuda:
                self.is_cuda = True
                x = x.cuda()
            self.writer.add_graph(self.model, x)
            del x

    def _log_engine_output(self, engine):
        """
        This function logs the engine.state.output if engine.state.output is a single scalar or a dictionary of scalars.
        """
        if engine.state.output is None:
            raise ValueError('If use_output is True, engine.state.output cannot be None.')
        else:
            if isinstance(engine.state.output, dict):
                for key, value in engine.state.output.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(''.join(['trainer/', key]),
                                               value,
                                               engine.state.iteration)
            elif isinstance(engine.state.output, (int, float)):
                self.writer.add_scalar('trainer/output',
                                       engine.state.output,
                                       engine.state.iteration)
            else:
                raise ValueError(
                    'Preferred format for engine.state.output is a single scalar or dictionary of scalars.')

    def _log_metrics(self, engine):
        if engine.state.metrics is None:
            raise ValueError('If use_metrics, engine.state.metrics cannot be None. Please attach metrics to engine.')
        else:
            for key, value in engine.state.metrics.items():
                self.writer.add_scalar(''.join(['trainer/', key]), value, engine.state.epoch)

    def _log_custom_state(self, engine):
        for key in self.state_keys:
            if hasattr(engine.state, key):
                state = getattr(engine.state, key)
                if isinstance(state, (int, float)):
                    self.writer.add_scalar(''.join(['trainer/', key]),
                                           getattr(engine.state, key),
                                           engine.state.iteration)
                else:
                    raise ValueError('engine.state.{key} should be of int or float.'.format(key=key))
            else:
                raise ValueError('engine.state does not have attribute %s.' % key)

    def _log_evaluator(self, engine, evaluator, mode='training'):
        if not isinstance(evaluator, Engine) and evaluator.state.metrics is not None:
            raise ValueError('evaluator must be an instance of ignite.Engine and have metrics attached to it.')
        else:
            for key, value in evaluator.state.metrics.items():
                self.writer.add_scalar(''.join([mode, '/', key]), value, engine.state.epoch)

    def _log_histograms(self, engine, model, add_name=False):
        if engine.state.epoch % self.histogram_freq == 0:
            for name, param in model.named_parameters():
                name = name.replace('.', '/')
                if add_name:
                    name = ''.join([model.__class__.__name__, '/', name])

                if param.requires_grad:
                    self.writer.add_histogram(name,
                                              param.cpu().data.numpy().flatten(),
                                              engine.state.epoch)
                    if self.write_grads:
                        self.writer.add_histogram(name + '_grad',
                                                  param.grad.cpu().data.numpy().flatten(),
                                                  engine.state.epoch)

    def _log_weight_image(self, engine, model):

        if engine.state.epoch % self.histogram_freq == 0:

            prefix = model.__class__.__name__ + '/'

            for name, param in model.named_parameters():

                name = ''.join([prefix, name.replace('.', '/')])

                w_img = param.data
                shape = list(w_img.size())

                if len(shape) == 1:
                    w_img = w_img.view(shape[0], 1, 1, 1)
                    grid = self._vistensor(w_img,
                                           ch=0,
                                           allkernels=True,
                                           padding=0,
                                           nrow=int(shape[0]/16) + 1)
                elif len(shape) == 2:
                    if shape[0] > shape[1]:
                        w_img = torch.transpose(w_img, 0, 1)
                        shape = list(w_img.size())
                    w_img = w_img.view(shape[0] * shape[1], 1, 1, 1)
                    grid = self._vistensor(w_img,
                                           ch=0,
                                           allkernels=True,
                                           nrow=shape[1],
                                           padding=0)
                elif len(shape) == 3:
                    w_img = w_img.view(shape[0], 1, shape[1], shape[2])
                    grid = self._vistensor(w_img,
                                           ch=0,
                                           allkernels=True,
                                           nrow=int(shape[0]/4) + 1,
                                           padding=1)
                elif len(shape) == 4:
                    grid = self._vistensor(w_img,
                                           ch=0,
                                           allkernels=True,
                                           nrow=int(shape[0]*shape[1]/4) + 1,
                                           padding=1)
                else:
                    continue

                self.writer.add_image(name, grid.numpy(), global_step=engine.state.epoch)

    def _base_epoch_functions(self, engine):
        # Use engine.state.metrics
        if self.use_metrics:
            self._log_metrics(engine=engine)

        # Train Evaluator
        if self.train_evaluator is not None:
            self._log_evaluator(engine=engine,
                                evaluator=self.train_evaluator,
                                mode='training')
        # Validation Evaluator
        if self.validation_evaluator is not None:
            self._log_evaluator(engine=engine,
                                evaluator=self.validation_evaluator,
                                mode='validation')

        # Histogram of Weights and Gradients; in the case of multiple models, users can create custom functions
        if self.model is not None and self.histogram_freq > 0:
            self._log_histograms(engine=engine,
                                 model=self.model,
                                 add_name=False)

        if self.write_images and self.model is not None and self.histogram_freq > 0:
            self._log_weight_image(engine=engine, model=self.model)

    def _base_iteration_functions(self, engine):
        # Use user created state attributes such as 'reward' for RL
        if self.state_keys is not None:
            self._log_custom_state(engine=engine)

        # Use engine.state.output for example mnist.py example
        if self.use_output:
            self._log_engine_output(engine=engine)

    def _on_complete(self, engine):
        self.writer.close()

    def _call_children_function(self):
        functions = [getattr(self, x)
                     for x in dir(self) if (not x.startswith('_') and
                                            type(getattr(self, x)) == MethodType and
                                            x != 'attach')]
        return functions

    def _vistensor(self, tensor, ch=0, allkernels=False, nrow=8, padding=1):
        '''
        vistensor: visuzlization tensor
            @ch: visualization channel
            @allkernels: visualization all tensors
        '''

        n, c, w, h = tensor.shape
        if allkernels:
            tensor = tensor.view(n * c, -1, w, h)
        elif c != 3:
            tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

        if w < 3 and h < 3:
            padding = 0

        rows = np.min((tensor.shape[0] // nrow + 1, 64))
        grid = utils.make_grid(tensor.cpu().data, nrow=rows, normalize=True, padding=padding)

        return grid

    def attach(self, engine):
        engine.add_event_handler(Events.STARTED, self._on_start)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._base_iteration_functions)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._base_epoch_functions)

        child_functions = self._call_children_function()
        for func in child_functions:
            engine.add_event_handler(Events.EPOCH_COMPLETED, func)

        engine.add_event_handler(Events.COMPLETED, self._on_complete)
