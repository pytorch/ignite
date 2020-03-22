import os
import tempfile
import numbers

from collections import namedtuple
import collections.abc as collections
import warnings

from typing import Optional, Callable, Mapping, Union

import torch

from ignite.engine import Events, Engine

__all__ = ["Checkpoint", "DiskSaver", "ModelCheckpoint"]


class Checkpoint:
    """Checkpoint handler can be used to periodically save and load objects which have attribute
    `state_dict`/`load_state_dict`. This class can use specific save handlers to store on the disk or a cloud
    storage, etc.

    Args:
        to_save (dict): Dictionary with the objects to save. Objects should have implemented `state_dict` and `
            load_state_dict` methods.
        save_handler (callable): Method to use to save engine and other provided objects. Function receives a
            checkpoint as a dictionary to save. In case if user needs to save engine's checkpoint on a disk,
            `save_handler` can be defined with :class:`~ignite.handlers.DiskSaver`.
        filename_prefix (str, optional): Prefix for the filename to which objects will be saved. See Note for details.
        score_function (callable, optional): If not None, it should be a function taking a single argument,
            :class:`~ignite.engine.Engine` object, and returning a score (`float`). Objects with highest scores will be
            retained.
        score_name (str, optional): If `score_function` not None, it is possible to store its value using
            `score_name`. See Notes for more details.
        n_saved (int, optional): Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`~ignite.handlers.global_step_from_engine`.
        archived (bool, optional): It True, saved checkpoint extension will be `.pth.tar`, Default value is False.

    Note:
        This class stores a single file as a dictionary of provided objects to save.
        The filename has the following structure: `{filename_prefix}_{name}_{suffix}.{ext}` where

        - `filename_prefix` is the argument passed to the constructor,
        - `name` is the key in `to_save` if a single object is to store, otherwise `name` is "checkpoint".
        - `ext` is `.pth.tar` if `archived=True` otherwise `.pth`.
        - `suffix` is composed as following `{global_step}_{score_name}={score}`.

        Above `global_step` defined by the output of `global_step_transform` and `score` defined by the output
        of `score_function`.

        By default, none of `score_function`, `score_name`, `global_step_transform` is defined, then suffix is
        setup by attached engine's current iteration. The filename will be
        `{filename_prefix}_{name}_{engine.state.iteration}.{ext}`.

        If defined a `score_function`, but without `score_name`, then suffix is defined by provided score.
        The filename will be `{filename_prefix}_{name}_{global_step}_{score}.pth`.

        If defined `score_function` and `score_name`, then the filename will
        be `{filename_prefix}_{name}_{score_name}={score}.{ext}`. If `global_step_transform` is provided, then
        the filename will be `{filename_prefix}_{name}_{global_step}_{score_name}={score}.{ext}`

        For example, `score_name="neg_val_loss"` and `score_function` that returns `-loss` (as objects with highest
        scores will be retained), then saved filename will be `{filename_prefix}_{name}_neg_val_loss=-0.1234.pth`.

        To get the last stored filename, handler exposes attribute `last_checkpoint`:

        .. code-block:: python

            handler = Checkpoint(...)
            ...
            print(handler.last_checkpoint)
            > checkpoint_12345.pth

    Examples:

        Attach the handler to make checkpoints during training:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Checkpoint, DiskSaver

            trainer = ...
            model = ...
            optimizer = ...
            lr_scheduler = ...

            to_save = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'trainer': trainer}
            handler = Checkpoint(to_save, DiskSaver('/tmp/models', create_dir=True), n_saved=2)
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)
            trainer.run(data_loader, max_epochs=6)
            > ["checkpoint_7000.pth", "checkpoint_8000.pth", ]

        Attach the handler to an evaluator to save best model during the training
        according to computed validation metric:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

            trainer = ...
            evaluator = ...
            # Setup Accuracy metric computation on evaluator
            # Run evaluation on epoch completed event
            # ...

            def score_function(engine):
                return engine.state.metrics['accuracy']

            to_save = {'model': model}
            handler = Checkpoint(to_save, DiskSaver('/tmp/models', create_dir=True), n_saved=2,
                                 filename_prefix='best', score_function=score_function, score_name="val_acc",
                                 global_step_transform=global_step_from_engine(trainer))

            evaluator.add_event_handler(Events.COMPLETED, handler)

            trainer.run(data_loader, max_epochs=10)
            > ["best_model_9_val_acc=0.77.pth", "best_model_10_val_acc=0.78.pth", ]

    """

    Item = namedtuple("Item", ["priority", "filename"])

    def __init__(
        self,
        to_save: Mapping,
        save_handler: Callable,
        filename_prefix: str = "",
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: Optional[int] = 1,
        global_step_transform: Callable = None,
        archived: bool = False,
    ):

        if not isinstance(to_save, collections.Mapping):
            raise TypeError("Argument `to_save` should be a dictionary, but given {}".format(type(to_save)))

        if len(to_save) < 1:
            raise ValueError("No objects to checkpoint.")

        if not callable(save_handler):
            raise TypeError("Argument `save_handler` should be callable")

        if score_function is None and score_name is not None:
            raise ValueError("If `score_name` is provided, then `score_function` " "should be also provided.")

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(
                "global_step_transform should be a function, got {} instead.".format(type(global_step_transform))
            )

        self._check_objects(to_save, "state_dict")
        self._fname_prefix = filename_prefix + "_" if len(filename_prefix) > 0 else filename_prefix
        self.save_handler = save_handler
        self.to_save = to_save
        self._score_function = score_function
        self._score_name = score_name
        self._n_saved = n_saved
        self._saved = []
        self._ext = ".pth.tar" if archived else ".pth"
        self.global_step_transform = global_step_transform

    @property
    def last_checkpoint(self) -> str:
        if len(self._saved) < 1:
            return None
        return self._saved[-1].filename

    def _check_lt_n_saved(self, or_equal=False):
        if self._n_saved is None:
            return True
        return len(self._saved) < self._n_saved + int(or_equal)

    def __call__(self, engine: Engine) -> None:

        suffix = ""
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, engine.last_event_name)
            suffix = "{}".format(global_step)

        if self._score_function is not None:
            priority = self._score_function(engine)
            if not isinstance(priority, numbers.Number):
                raise ValueError("Output of score_function should be a number")
        else:
            priority = engine.state.get_event_attrib_value(Events.ITERATION_COMPLETED)

        if self._check_lt_n_saved() or self._saved[0].priority < priority:

            priority_str = (
                "{}".format(priority) if isinstance(priority, numbers.Integral) else "{:.4f}".format(priority)
            )

            if self._score_name is not None:
                if len(suffix) > 0:
                    suffix += "_"
                suffix = "{}{}={}".format(suffix, self._score_name, priority_str)
            elif self._score_function is not None:
                if len(suffix) > 0:
                    suffix += "_"
                suffix = "{}{}".format(suffix, priority_str)
            elif len(suffix) == 0:
                suffix = "{}".format(priority_str)

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
                checkpoint = checkpoint[name]
            filename = "{}{}_{}{}".format(self._fname_prefix, name, suffix, self._ext)

            if any(item.filename == filename for item in self._saved):
                return

            self.save_handler(checkpoint, filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda item: item[0])

        if not self._check_lt_n_saved(or_equal=True):
            item = self._saved.pop(0)
            self.save_handler.remove(item.filename)

    def _setup_checkpoint(self) -> dict:
        checkpoint = {}
        for k, obj in self.to_save.items():
            checkpoint[k] = obj.state_dict()
        return checkpoint

    @staticmethod
    def _check_objects(objs: Mapping, attr: str) -> None:
        for k, obj in objs.items():
            if not hasattr(obj, attr):
                raise TypeError("Object {} should have `{}` method".format(type(obj), attr))

    @staticmethod
    def load_objects(to_load: Mapping, checkpoint: Mapping) -> None:
        """Helper method to apply `load_state_dict` on the objects from `to_load` using states from `checkpoint`.

        Args:
            to_load (Mapping): a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint (Mapping): a dictionary with state_dicts to load, e.g. `{"model": model_state_dict,
                "optimizer": opt_state_dict}`. If `to_load` contains a single key, then checkpoint can contain directly
                corresponding state_dict.
        """
        Checkpoint._check_objects(to_load, "load_state_dict")
        if not isinstance(checkpoint, collections.Mapping):
            raise TypeError("Argument checkpoint should be a dictionary, but given {}".format(type(checkpoint)))
        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint:
                obj.load_state_dict(checkpoint)
                return

        # multiple objects to load
        for k, obj in to_load.items():
            if k not in checkpoint:
                raise ValueError("Object labeled by '{}' from `to_load` is not found in the checkpoint".format(k))
            obj.load_state_dict(checkpoint[k])


class DiskSaver:
    """Handler that saves input checkpoint on a disk.

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occures during saving).
        create_dir (bool, optional): if True, will create directory 'dirname' if it doesnt exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the directory 'dirname'.
    """

    def __init__(self, dirname: str, atomic: bool = True, create_dir: bool = True, require_empty: bool = True):
        self.dirname = os.path.expanduser(dirname)
        self._atomic = atomic
        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # Ensure that dirname exists
        if not os.path.exists(dirname):
            raise ValueError("Directory path '{}' is not found".format(dirname))

        if require_empty:
            matched = [fname for fname in os.listdir(dirname) if fname.endswith(".pth") or fname.endswith(".pth.tar")]
            if len(matched) > 0:
                raise ValueError(
                    "Files {} with extension '.pth' or '.pth.tar' are already present "
                    "in the directory {}. If you want to use this "
                    "directory anyway, pass `require_empty=False`."
                    "".format(matched, dirname)
                )

    def __call__(self, checkpoint: Mapping, filename: str) -> None:
        path = os.path.join(self.dirname, filename)

        if not self._atomic:
            torch.save(checkpoint, path)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            try:
                torch.save(checkpoint, tmp.file)
            except BaseException:
                tmp.close()
                os.remove(tmp.name)
                raise
            else:
                tmp.close()
                os.rename(tmp.name, path)

    def remove(self, filename: str) -> None:
        path = os.path.join(self.dirname, filename)
        os.remove(path)


class ModelCheckpoint(Checkpoint):
    """ModelCheckpoint handler can be used to periodically save objects to disk only. If needed to store checkpoints to
    another storage type, please consider :class:`~ignite.handlers.checkpoint.Checkpoint`.

    This handler expects two arguments:

        - an :class:`~ignite.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.

    See Examples for further details.

    .. warning::

        Behaviour of this class has been changed since v0.3.0.

        Argument `save_as_state_dict` is deprecated and should not be used. It is considered as True.

        Argument `save_interval` is deprecated and should not be used. Please, use events filtering instead, e.g.
        :attr:`~ignite.engine.Events.ITERATION_STARTED(every=1000)`

        There is no more internal counter that has been used to indicate the number of save actions. User could
        see its value `step_number` in the filename, e.g. `{filename_prefix}_{name}_{step_number}.pth`. Actually,
        `step_number` is replaced by current engine's epoch if `score_function` is specified and current iteration
        otherwise.

        A single `pth` file is created instead of multiple files.

    Args:
        dirname (str): Directory path where objects will be saved.
        filename_prefix (str): Prefix for the filenames to which objects will be saved. See Notes of
            :class:`~ignite.handlers.Checkpoint` for more details.
        score_function (callable, optional): if not None, it should be a function taking a single argument, an
            :class:`~ignite.engine.Engine` object, and return a score (`float`). Objects with highest scores will be
            retained.
        score_name (str, optional): if `score_function` not None, it is possible to store its value using
            `score_name`. See Notes for more details.
        n_saved (int, optional): Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        atomic (bool, optional): If True, objects are serialized to a temporary file, and then moved to final
            destination, so that files are guaranteed to not be damaged (for example if exception
            occurs during saving).
        require_empty (bool, optional): If True, will raise exception if there are any files starting with
            `filename_prefix` in the directory 'dirname'.
        create_dir (bool, optional): If True, will create directory 'dirname' if it doesnt exist.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`~ignite.handlers.global_step_from_engine`.
        archived (bool, optional): It True, saved checkpoint extension will be `.pth.tar`, Default value is False.

    Examples:
        >>> import os
        >>> from ignite.engine import Engine, Events
        >>> from ignite.handlers import ModelCheckpoint
        >>> from torch import nn
        >>> trainer = Engine(lambda batch: None)
        >>> handler = ModelCheckpoint('/tmp/models', 'myprefix', n_saved=2, create_dir=True)
        >>> model = nn.Linear(3, 3)
        >>> trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, {'mymodel': model})
        >>> trainer.run([0], max_epochs=6)
        >>> os.listdir('/tmp/models')
        ['myprefix_mymodel_4.pth', 'myprefix_mymodel_6.pth']
        >>> handler.last_checkpoint
        ['/tmp/models/myprefix_mymodel_6.pth']
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str,
        save_interval: Optional[Callable] = None,
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: int = 1,
        atomic: bool = True,
        require_empty: bool = True,
        create_dir: bool = True,
        save_as_state_dict: bool = True,
        global_step_transform: Optional[Callable] = None,
        archived: bool = False,
    ):

        if not save_as_state_dict:
            raise ValueError("Argument save_as_state_dict is deprecated and should be True")
        if save_interval is not None:
            msg = (
                "Argument save_interval is deprecated and should be None. "
                "Please, use events filtering instead, e.g. Events.ITERATION_STARTED(every=1000)"
            )
            if save_interval == 1:
                # Do not break for old version who used `save_interval=1`
                warnings.warn(msg)
            else:
                # No choice
                raise ValueError(msg)

        disk_saver = DiskSaver(dirname, atomic=atomic, create_dir=create_dir, require_empty=require_empty)

        if score_function is None and score_name is not None:
            raise ValueError("If `score_name` is provided, then `score_function` " "should be also provided.")

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(
                "global_step_transform should be a function, got {} instead.".format(type(global_step_transform))
            )

        self._fname_prefix = filename_prefix + "_" if len(filename_prefix) > 0 else filename_prefix
        self.save_handler = disk_saver
        self.to_save = None
        self._score_function = score_function
        self._score_name = score_name
        self._n_saved = n_saved
        self._saved = []
        self._ext = ".pth.tar" if archived else ".pth"
        self.global_step_transform = global_step_transform

    @property
    def last_checkpoint(self) -> Union[str, None]:
        if len(self._saved) < 1:
            return None
        return os.path.join(self.save_handler.dirname, self._saved[-1].filename)

    def __call__(self, engine: Engine, to_save: Mapping) -> None:

        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._check_objects(to_save, "state_dict")
        self.to_save = to_save
        super(ModelCheckpoint, self).__call__(engine)
