import os
import tempfile
from collections import namedtuple

import torch


class ModelCheckpoint:
    """ Foo bar

    Parameters
    ----------
    dirname : str
    filename_prefix : str
    iteration_interval : int, optional
    epoch_interval : int, optional
    score_function : Callable, optional
    n_saved : int, optional
    atomic : bool, optional
    require_empty : bool, optional
    create_dir : bool, optional
    exist_ok : bool, optional
    """

    def __init__(self, dirname, filename_prefix,
                 iteration_interval=None, epoch_interval=None,
                 score_function=None,
                 n_saved=1,
                 atomic=True, require_empty=True,
                 create_dir=True, exist_ok=True):

        self._dirname            = dirname
        self._fname_prefix       = filename_prefix
        self._n_saved            = n_saved
        self._iteration_interval = iteration_interval
        self._epoch_interval     = epoch_interval
        self._score_function     = score_function
        self._atomic             = atomic
        self._item_T             = namedtuple('item_T', ('priority', 'data'))

        self._saved = []

        if not (iteration_interval or epoch_interval or score_function):
            raise ValueError("One of 'iteration_interval', 'epoch_interval' or "
                             "'score_function' arguments must be provided.")

        if create_dir:
            os.makedirs(dirname, exist_ok=exist_ok)

        if require_empty:
            n_matched = [fname
                         for fname in os.listdir(dirname)
                         if fname.startswith(self._fname_prefix)]

            if len(n_matched) > 0:
                raise ValueError("Files prefixed with {} are already present "
                                 "in the directory {}. If you want to use this "
                                 "directory anyway, pass require_empty=False. "
                                 "".format(filename_prefix, dirname))

    def _save(self, obj, path):
        if not self._atomic:
            torch.save(obj, path)

        else:
            tmp = tempfile.NamedTemporaryFile(delete=False)

            try:
                torch.save(obj, tmp.file)

            except:
                tmp.close()
                os.remove(tmp.name)
                raise

            else:
                tmp.close()
                os.rename(tmp.name, path)

    def __call__(self, engine, **kwargs):

        if len(kwargs) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        if self._score_function is not None:
            index    = engine.current_iteration
            priority = self._score_function(engine)

        elif self._iteration_interval is not None:
            if engine.current_iteration % self._iteration_interval != 0:
                return
            index    = engine.current_iteration
            priority = engine.current_iteration

        else:
            if engine.current_epoch % self._epoch_interval != 0:
                return
            index    = engine.current_epoch
            priority = engine.current_epoch

        if (len(self._saved) < self._n_saved) or (self._saved[0].priority < priority):
            saved = []

            for name, obj in kwargs.items():
                fname = '{}_{}_{}.pth'.format(self._fname_prefix, name, index)
                path  = os.path.join(self._dirname, fname)

                self._save(obj=obj, path=path)
                saved.append(fname)

            saved = self._item_T(priority, saved)
            self._saved.append(saved)
            self._saved.sort(key=lambda item: item.priority)

        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)
