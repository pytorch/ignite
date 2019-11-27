import os
import tempfile
from collections import namedtuple
import warnings

import sys

IS_PYTHON2 = sys.version_info[0] < 3

if IS_PYTHON2:
    import collections
else:
    import collections.abc as collections

import torch


class Checkpoint(object):
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
            retained. Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        score_name (str, optional): If `score_function` not None, it is possible to store its absolute value using
            `score_name`. See Notes for more details.
        n_saved (int, optional): Number of objects that should be kept on disk. Older files will be removed.

    Note:
        This class stores a single file as a dictionary of provided objects to save.
        The filename has the following structure: `{filename_prefix}_{name}_{suffix}.pth` where

        - `filename_prefix` is the argument passed to the constructor,
        - `name` is the key in `to_save` if a single object is to store, otherwise `name` is "checkpoint".
        - `suffix` can have 3 possible values:

        1) Default, when no `score_function`/`score_name` defined: Suffix is defined by attached engine's current
        iteration. The filename will be `{filename_prefix}_{name}_{engine.state.iteration}.pth`.

        2) With `score_function`, but without `score_name`: Suffix is defined by provided score. The filename will be
        `{filename_prefix}_{name}_{score}.pth`.

        3) With `score_function` and `score_name`: In this case, user can store its absolute value using `score_name`
        in the filename. Suffix is defined by engine's current epoch, score name and its value. The filename will
        be `{filename_prefix}_{name}_{engine.state.epoch}_{score_name}={abs(score)}.pth`.
        For example, `score_name="val_loss"` and `score_function` that returns `-loss` (as objects with
        highest scores will be retained), then saved models filenames will be `model_0_val_loss=0.1234.pth`.

    Examples:
        Saved checkpoint then can be used to resume a training:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Checkpoint, DiskSaver
            from torch import nn
            trainer = Engine(lambda batch: None)
            model = ...
            optimizer = ...
            lr_scheduler = ...
            to_save = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
            handler = Checkpoint(to_save, DiskSaver('/tmp/models', create_dir=True), 'myprefix', n_saved=2)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
            trainer.run([0], max_epochs=6)
            os.listdir('/tmp/models')


    """

    Item = namedtuple("Item", ["priority", "filename"])

    def __init__(self, to_save, save_handler, filename_prefix="",
                 score_function=None, score_name=None,
                 n_saved=1):

        if not isinstance(to_save, collections.Mapping):
            raise TypeError("Argument `to_save` should be a dictionary, but given {}".format(type(to_save)))

        if len(to_save) < 1:
            raise ValueError("No objects to checkpoint.")

        if not callable(save_handler):
            raise TypeError("Argument `save_handler` should be callable")

        if score_function is None and score_name is not None:
            raise ValueError("If `score_name` is provided, then `score_function` "
                             "should be also provided.")

        self._check_objects(to_save)
        self._fname_prefix = filename_prefix + "_" if len(filename_prefix) > 0 else filename_prefix
        self.save_handler = save_handler
        self.to_save = to_save
        self._score_function = score_function
        self._score_name = score_name
        self._n_saved = n_saved
        self._saved = []

    @property
    def last_checkpoint(self):
        if len(self._saved) < 1:
            return None
        return self._saved[0].filename

    def __call__(self, engine):

        if self._score_function is not None:
            priority = self._score_function(engine)
        else:
            priority = engine.state.iteration

        if len(self._saved) < self._n_saved or \
                self._saved[0].priority < priority:

            suffix = "{}".format(priority)
            if self._score_name is not None:
                suffix = "{}_{}={:.7}".format(engine.state.epoch, self._score_name, abs(priority))

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
            filename = '{}{}_{}.pth'.format(self._fname_prefix, name, suffix)

            self.save_handler(checkpoint, filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda item: item[0])

        if len(self._saved) > self._n_saved:
            item = self._saved.pop(0)
            self.save_handler.remove(item.filename)

    def _setup_checkpoint(self):
        checkpoint = {}
        for k, obj in self.to_save.items():
            checkpoint[k] = obj.state_dict()
        return checkpoint

    @staticmethod
    def _check_objects(to_save_or_load):
        for k, obj in to_save_or_load.items():
            if not (hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict")):
                raise TypeError("Object {} should have `state_dict` and `load_state_dict` methods".format(type(obj)))

    @staticmethod
    def load_objects(to_load, checkpoint):
        """Method to apply `load_state_dict` on the objects from `to_load` using states from `checkpoint`.

        Args:
            to_load (Mapping):
            checkpoint (Mapping):
        """
        Checkpoint._check_objects(to_load)
        if not isinstance(checkpoint, collections.Mapping):
            raise TypeError("Argument checkpoint should be a dictionary, but given {}".format(type(checkpoint)))
        for k, obj in to_load.items():
            if k not in checkpoint:
                raise ValueError("Object labeled by '{}' from `to_load` is not found in the checkpoint".format(k))
            obj.load_state_dict(checkpoint[k])


class DiskSaver(object):
    """Handler that saves input checkpoint on a disk.

    Args:
        dirname (str): Directory path where the checkpoint will be saved
        atomic (bool, optional): if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occures during saving).
        create_dir (bool, optional): if True, will create directory 'dirname' if it doesnt exist.
        require_empty (bool, optional): If True, will raise exception if there are any files in the directory 'dirname'.

    """

    def __init__(self, dirname, atomic=True, create_dir=True, require_empty=True):
        self.dirname = os.path.expanduser(dirname)
        self._atomic = atomic
        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # Ensure that dirname exists
        if not os.path.exists(dirname):
            raise ValueError("Directory path '{}' is not found".format(dirname))

        if require_empty:
            matched = [fname for fname in os.listdir(dirname)]
            if len(matched) > 0:
                raise ValueError("Files are already present "
                                 "in the directory {}. If you want to use this "
                                 "directory anyway, pass `require_empty=False`."
                                 "".format(dirname))

    def __call__(self, checkpoint, filename):
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

    def remove(self, filename):
        path = os.path.join(self.dirname, filename)
        os.remove(path)


class ModelCheckpoint(Checkpoint):
    """ModelCheckpoint handler can be used to periodically save objects to disk.

    This handler expects two arguments:

        - an :class:`~ignite.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.

    See Notes and Examples for further details.

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
        dirname (str):
            Directory path where objects will be saved.
        filename_prefix (str):
            Prefix for the filenames to which objects will be saved. See Notes of :class:`~ignite.handlers.Checkpoint`
            for more details.
        score_function (callable, optional):
            if not None, it should be a function taking a single argument,
            an :class:`~ignite.engine.Engine` object,
            and return a score (`float`). Objects with highest scores will be retained.
            Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        score_name (str, optional):
            if `score_function` not None, it is possible to store its absolute value using `score_name`. See Notes for
            more details.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
        atomic (bool, optional):
            If True, objects are serialized to a temporary file,
            and then moved to final destination, so that files are
            guaranteed to not be damaged (for example if exception occures during saving).
        require_empty (bool, optional):
            If True, will raise exception if there are any files starting with `filename_prefix`
            in the directory 'dirname'.
        create_dir (bool, optional):
            If True, will create directory 'dirname' if it doesnt exist.

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
    """

    def __init__(self, dirname, filename_prefix,
                 save_interval=None,
                 score_function=None, score_name=None,
                 n_saved=1,
                 atomic=True, require_empty=True,
                 create_dir=True,
                 save_as_state_dict=True):

        if not save_as_state_dict:
            raise ValueError("Argument save_as_state_dict is deprecated and should be True")
        if save_interval is not None:
            msg = "Argument save_interval is deprecated and should be None. " \
                  "Please, use events filtering instead, e.g. Events.ITERATION_STARTED(every=1000)"
            if save_interval == 1:
                # Do not break for old version who used `save_interval=1`
                warnings.warn(msg)
            else:
                # No choice
                raise ValueError(msg)

        disk_saver = DiskSaver(dirname, atomic=atomic, create_dir=create_dir, require_empty=require_empty)

        if score_function is None and score_name is not None:
            raise ValueError("If `score_name` is provided, then `score_function` "
                             "should be also provided.")

        self._fname_prefix = filename_prefix + "_" if len(filename_prefix) > 0 else filename_prefix
        self.save_handler = disk_saver
        self.to_save = None
        self._score_function = score_function
        self._score_name = score_name
        self._n_saved = n_saved
        self._saved = []

    @property
    def last_checkpoint(self):
        if len(self._saved) < 1:
            return None
        return os.path.join(self.save_handler.dirname, self._saved[0].filename)

    def __call__(self, engine, to_save):

        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._check_objects(to_save)
        self.to_save = to_save
        super(ModelCheckpoint, self).__call__(engine)
