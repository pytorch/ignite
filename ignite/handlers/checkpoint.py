import collections.abc as collections
import numbers
import os
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Callable, Mapping, Optional, Union

import torch
import torch.nn as nn

import ignite.distributed as idist
from ignite.engine import Engine, Events

__all__ = ["Checkpoint", "DiskSaver", "ModelCheckpoint", "BaseSaveHandler"]


class BaseSaveHandler(metaclass=ABCMeta):
    """Base class for save handlers

    Methods to override:

    - :meth:`~ignite.handlers.checkpoint.BaseSaveHandler.__call__`
    - :meth:`~ignite.handlers.checkpoint.BaseSaveHandler.remove`


    Note:
        In derived class, please, make sure that in distributed configuration overridden methods are called by a single
        process. Distributed configuration on XLA devices should be treated slightly differently: for saving checkpoint
        with `xm.save() <https://pytorch.org/xla/release/1.5/index.html#torch_xla.core.xla_model.save>`_  all processes
        should pass into the function. Otherwise, application gets stuck.

    """

    @abstractmethod
    def __call__(self, checkpoint: Mapping, filename: str, metadata: Optional[Mapping] = None) -> None:
        """Method to save `checkpoint` with `filename`. Additionally, metadata dictionary is provided.

        Metadata contains:

        - `basename`: file prefix (if provided) with checkpoint name, e.g. `epoch_checkpoint`.
        - `score_name`: score name if provided, e.g `val_acc`.
        - `priority`: checkpoint priority value (higher is better), e.g. `12` or `0.6554435`

        Args:
            checkpoint (Mapping): checkpoint dictionary to save.
            filename (str): filename associated with checkpoint.
            metadata (Mapping, optional): metadata on checkpoint to save.

        """
        pass

    @abstractmethod
    def remove(self, filename: str) -> None:
        """Method to remove saved checkpoint.

        Args:
            filename (str): filename associated with checkpoint.

        """
        pass


class Checkpoint:
    """Checkpoint handler can be used to periodically save and load objects which have attribute
    `state_dict`/`load_state_dict`. This class can use specific save handlers to store on the disk or a cloud
    storage, etc. The Checkpoint handler (if used with :class:`~ignite.handlers.DiskSaver`) also handles automatically
    moving data on TPU to CPU before writing the checkpoint.

    Args:
        to_save (Mapping): Dictionary with the objects to save. Objects should have implemented `state_dict` and `
            load_state_dict` methods. If contains objects of type torch `DistributedDataParallel`_ or
            `DataParallel`_, their internal wrapped model is automatically saved (to avoid additional key ``module.`` in
            the state dictionary).
        save_handler (callable or :class:`~ignite.handlers.checkpoint.BaseSaveHandler`): Method or callable class to
            use to save engine and other provided objects. Function receives two objects: checkpoint as a dictionary
            and filename. If `save_handler` is callable class, it can
            inherit of :class:`~ignite.handlers.checkpoint.BaseSaveHandler` and optionally implement `remove` method to
            keep a fixed number of saved checkpoints. In case if user needs to save engine's checkpoint on a disk,
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
        archived (bool, optional): Deprecated argument as models saved by `torch.save` are already compressed.

    .. _DistributedDataParallel: https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel
    .. _DataParallel: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel

    Note:
        This class stores a single file as a dictionary of provided objects to save.
        The filename has the following structure: `{filename_prefix}_{name}_{suffix}.{ext}` where

        - `filename_prefix` is the argument passed to the constructor,
        - `name` is the key in `to_save` if a single object is to store, otherwise `name` is "checkpoint".
        - `suffix` is composed as following `{global_step}_{score_name}={score}`.

        Above `global_step` defined by the output of `global_step_transform` and `score` defined by the output
        of `score_function`.

        By default, none of `score_function`, `score_name`, `global_step_transform` is defined, then suffix is
        setup by attached engine's current iteration. The filename will be
        `{filename_prefix}_{name}_{engine.state.iteration}.{ext}`.

        If defined a `score_function`, but without `score_name`, then suffix is defined by provided score.
        The filename will be `{filename_prefix}_{name}_{global_step}_{score}.pt`.

        If defined `score_function` and `score_name`, then the filename will
        be `{filename_prefix}_{name}_{score_name}={score}.{ext}`. If `global_step_transform` is provided, then
        the filename will be `{filename_prefix}_{name}_{global_step}_{score_name}={score}.{ext}`

        For example, `score_name="neg_val_loss"` and `score_function` that returns `-loss` (as objects with highest
        scores will be retained), then saved filename will be `{filename_prefix}_{name}_neg_val_loss=-0.1234.pt`.

        To get the last stored filename, handler exposes attribute `last_checkpoint`:

        .. code-block:: python

            handler = Checkpoint(...)
            ...
            print(handler.last_checkpoint)
            > checkpoint_12345.pt

    Note:
        This class is distributed configuration-friendly: it is not required to instantiate the class in rank 0 only
        process. This class supports automatically distributed configuration and if used with
        :class:`~ignite.handlers.DiskSaver`, checkpoint is stored by rank 0 process.

    .. warning::

        When running on TPUs, it should be run in all processes, otherwise application can get stuck on saving the
        checkpoint.

        .. code-block:: python

            # Wrong:
            # if idist.get_rank() == 0:
            #     handler = Checkpoint(...)
            #     trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)

            # Correct:
            handler = Checkpoint(...)
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)

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
            > ["checkpoint_7000.pt", "checkpoint_8000.pt", ]

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
            > ["best_model_9_val_acc=0.77.pt", "best_model_10_val_acc=0.78.pt", ]

    """

    Item = namedtuple("Item", ["priority", "filename"])

    def __init__(
        self,
        to_save: Mapping,
        save_handler: Union[Callable, BaseSaveHandler],
        filename_prefix: str = "",
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: Optional[int] = 1,
        global_step_transform: Callable = None,
        archived: bool = False,
    ):

        if to_save is not None:  # for compatibility with ModelCheckpoint
            if not isinstance(to_save, collections.Mapping):
                raise TypeError("Argument `to_save` should be a dictionary, but given {}".format(type(to_save)))

            if len(to_save) < 1:
                raise ValueError("No objects to checkpoint.")

            self._check_objects(to_save, "state_dict")

        if not (callable(save_handler) or isinstance(save_handler, BaseSaveHandler)):
            raise TypeError("Argument `save_handler` should be callable or inherit from BaseSaveHandler")

        if score_function is None and score_name is not None:
            raise ValueError("If `score_name` is provided, then `score_function` " "should be also provided.")

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(
                "global_step_transform should be a function, got {} instead.".format(type(global_step_transform))
            )
        if archived:
            warnings.warn("Argument archived is deprecated and will be removed in 0.5.0")

        self.to_save = to_save
        self._fname_prefix = filename_prefix + "_" if len(filename_prefix) > 0 else filename_prefix
        self.save_handler = save_handler
        self._score_function = score_function
        self._score_name = score_name
        self._n_saved = n_saved
        self._saved = []
        self._ext = ".pt"
        self.global_step_transform = global_step_transform

    @property
    def last_checkpoint(self) -> Optional[str]:
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

            metadata = {
                "basename": "{}{}".format(self._fname_prefix, name),
                "score_name": self._score_name,
                "priority": priority,
            }

            try:
                self.save_handler(checkpoint, filename, metadata)
            except TypeError:
                self.save_handler(checkpoint, filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda item: item[0])

        if not self._check_lt_n_saved(or_equal=True):
            item = self._saved.pop(0)
            if isinstance(self.save_handler, BaseSaveHandler):
                self.save_handler.remove(item.filename)

    def _setup_checkpoint(self) -> dict:
        checkpoint = {}
        for k, obj in self.to_save.items():
            if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                obj = obj.module
            checkpoint[k] = obj.state_dict()
        return checkpoint

    @staticmethod
    def _check_objects(objs: Mapping, attr: str) -> None:
        for k, obj in objs.items():
            if not hasattr(obj, attr):
                raise TypeError("Object {} should have `{}` method".format(type(obj), attr))

    @staticmethod
    def load_objects(to_load: Mapping, checkpoint: Mapping, **kwargs) -> None:
        """Helper method to apply `load_state_dict` on the objects from `to_load` using states from `checkpoint`.

        Exemples:

        .. code-block:: python

            import torch
            from ignite.engine import Engine, Events
            from ignite.handlers import ModelCheckpoint, Checkpoint
            trainer = Engine(lambda engine, batch: None)
            handler = ModelCheckpoint('/tmp/models', 'myprefix', n_saved=None, create_dir=True)
            model = torch.nn.Linear(3, 3)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            to_save = {"weights": model, "optimizer": optimizer}
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, to_save)
            trainer.run(torch.randn(10, 1), 5)

            to_load = to_save
            checkpoint_fp = "/tmp/models/myprefix_checkpoint_40.pth"
            checkpoint = torch.load(checkpoint_fp)
            Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        Args:
            to_load (Mapping): a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint (Mapping): a dictionary with state_dicts to load, e.g. `{"model": model_state_dict,
                "optimizer": opt_state_dict}`. If `to_load` contains a single key, then checkpoint can contain directly
                corresponding state_dict.
            **kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)
        """
        Checkpoint._check_objects(to_load, "load_state_dict")
        if not isinstance(checkpoint, collections.Mapping):
            raise TypeError("Argument checkpoint should be a dictionary, but given {}".format(type(checkpoint)))

        if len(kwargs) > 1 or any(k for k in kwargs.keys() if k not in ["strict"]):
            warnings.warn("kwargs contains keys other than strict and these will be ignored")

        is_state_dict_strict = kwargs.get("strict", True)
        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint:
                obj.load_state_dict(checkpoint, strict=is_state_dict_strict)
                return

        # multiple objects to load
        for k, obj in to_load.items():
            if k not in checkpoint:
                raise ValueError("Object labeled by '{}' from `to_load` is not found in the checkpoint".format(k))
            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(checkpoint[k], strict=is_state_dict_strict)
            else:
                obj.load_state_dict(checkpoint[k])


class DiskSaver(BaseSaveHandler):
    """Handler that saves input checkpoint on a disk.

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occurs during saving).
        create_dir (bool, optional): if True, will create directory 'dirname' if it doesnt exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the directory 'dirname'.
    """

    def __init__(
        self, dirname: str, atomic: bool = True, create_dir: bool = True, require_empty: bool = True,
    ):
        self.dirname = os.path.expanduser(dirname)
        self._atomic = atomic
        self._check_and_setup(dirname, create_dir, require_empty)

    @staticmethod
    @idist.one_rank_only()
    def _check_and_setup(dirname, create_dir, require_empty):
        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # Ensure that dirname exists
        if not os.path.exists(dirname):
            raise ValueError("Directory path '{}' is not found".format(dirname))

        if require_empty:
            matched = [fname for fname in os.listdir(dirname) if fname.endswith(".pt")]
            if len(matched) > 0:
                raise ValueError(
                    "Files {} with extension '.pt' are already present "
                    "in the directory {}. If you want to use this "
                    "directory anyway, pass `require_empty=False`."
                    "".format(matched, dirname)
                )

    def __call__(self, checkpoint: Mapping, filename: str, metadata: Optional[Mapping] = None) -> None:
        path = os.path.join(self.dirname, filename)

        if idist.has_xla_support:
            self._save_xla(checkpoint, path)
        else:
            self._save_native(checkpoint, path)

    @idist.one_rank_only()
    def _save_native(self, checkpoint: Mapping, path: str):
        self._save_func(checkpoint, path, torch.save)

    def _save_xla(self, checkpoint: Mapping, path: str):
        import torch_xla.core.xla_model as xm

        # all tpu procs should enter here as internally performs sync across device
        self._save_func(checkpoint, path, xm.save, rank=idist.get_rank())

    def _save_func(self, checkpoint: Mapping, path: str, func: Callable, rank: int = 0):
        if not self._atomic:
            func(checkpoint, path)
        else:
            tmp_file = None
            tmp_name = None
            tmp = None
            if rank == 0:
                tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
                tmp_file = tmp.file
                tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file)
            except BaseException:
                if tmp is not None:
                    tmp.close()
                    os.remove(tmp_name)
                    raise
            else:
                if tmp is not None:
                    tmp.close()
                    os.rename(tmp.name, path)

    @idist.one_rank_only()
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
        see its value `step_number` in the filename, e.g. `{filename_prefix}_{name}_{step_number}.pt`. Actually,
        `step_number` is replaced by current engine's epoch if `score_function` is specified and current iteration
        otherwise.

        A single `pt` file is created instead of multiple files.

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
        archived (bool, optional): Deprecated argument as models saved by `torch.save` are already compressed.

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
        ['myprefix_mymodel_4.pt', 'myprefix_mymodel_6.pt']
        >>> handler.last_checkpoint
        ['/tmp/models/myprefix_mymodel_6.pt']
    """

    def __init__(
        self,
        dirname: str,
        filename_prefix: str,
        save_interval: Optional[Callable] = None,
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: Union[int, None] = 1,
        atomic: bool = True,
        require_empty: bool = True,
        create_dir: bool = True,
        save_as_state_dict: bool = True,
        global_step_transform: Optional[Callable] = None,
        archived: bool = False,
    ):

        if not save_as_state_dict:
            raise ValueError(
                "Argument save_as_state_dict is deprecated and should be True."
                "This argument will be removed in 0.5.0."
            )
        if save_interval is not None:
            msg = (
                "Argument save_interval is deprecated and should be None. This argument will be removed in 0.5.0."
                "Please, use events filtering instead, e.g. Events.ITERATION_STARTED(every=1000)"
            )
            if save_interval == 1:
                # Do not break for old version who used `save_interval=1`
                warnings.warn(msg)
            else:
                # No choice
                raise ValueError(msg)

        disk_saver = DiskSaver(dirname, atomic=atomic, create_dir=create_dir, require_empty=require_empty,)

        super(ModelCheckpoint, self).__init__(
            to_save=None,
            save_handler=disk_saver,
            filename_prefix=filename_prefix,
            score_function=score_function,
            score_name=score_name,
            n_saved=n_saved,
            global_step_transform=global_step_transform,
            archived=archived,
        )

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
