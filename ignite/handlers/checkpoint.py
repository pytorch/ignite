import os
import tempfile

import torch


class ModelCheckpoint(object):
    """ ModelCheckpoint handler can be used to periodically save objects to disk.

    This handler accepts two arguments:
        - an `ignite.engines.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.
            See Notes and Examples for further details.

    Args:
        dirname (str):
            Directory path where objects will be saved
        filename_prefix (str):
            Prefix for the filenames to which objects will be saved. See Notes
            for more details.
        save_interval (int, optional):
            if not None, objects will be saved to disk every `save_interval` calls to the handler.
            Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        score_function (Callable, optional):
            if not None, it should be a function taking a single argument,
            an `ignite.engines.Engine` object,
            and return a score (`float`). Objects with highest scores will be retained.
            Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
        atomic (bool, optional):
            If True, objects are serialized to a temporary file,
            and then moved to final destination, so that files are
            guaranteed to not be damaged (for example if exception occures during saving).
        require_empty (bool, optional):
            If True, will raise exception if there are any files starting with `filename_prefix`
            in the directory 'dirname'
        create_dir (bool, optional):
            If True, will create directory 'dirname' if it doesnt exist.
        exist_ok (bool, optional):
            Passed to 'os.makedirs' call. Ignored if 'create_dir' is False.

    Notes:
          This handler expects two arguments: an `Engine` object and a `dict`
          mapping names to objects that should be saved.

          These names are used to specify filenames for saved objects.
          Each filename has the following structure:
          `{filename_prefix}_{name}_{step_number}.pth`.
          Here, `filename_prefix` is the argument passed to the constructor,
          `name` is the key in the aforementioned `dict`, and `step_number`
          is incremented by `1` with every call to the handler.

    Examples:
        >>> import os
        >>> from ignite.engines import Engine, Events
        >>> from ignite.handlers import ModelCheckpoint
        >>> from torch import nn
        >>> trainer = Engine(lambda batch: None)
        >>> handler = ModelCheckpoint('/tmp/models', 'myprefix', save_interval=2, n_saved=2, create_dir=True)
        >>> model = nn.Linear(3, 3)
        >>> trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})
        >>> trainer.run([0], max_epochs=6)
        >>> os.listdir('/tmp/models')
        ['myprefix_mymodel_4.pth', 'myprefix_mymodel_6.pth']
    """

    def __init__(self, dirname, filename_prefix,
                 save_interval=None, score_function=None,
                 n_saved=1,
                 atomic=True, require_empty=True,
                 create_dir=True, exist_ok=False):

        self._dirname = dirname
        self._fname_prefix = filename_prefix
        self._n_saved = n_saved
        self._save_interval = save_interval
        self._score_function = score_function
        self._atomic = atomic
        self._saved = []  # list of tuples (priority, saved_objects)
        self._iteration = 0

        if not (save_interval is None) ^ (score_function is None):
            raise ValueError("Exactly one of `save_interval`, or `score_function` "
                             "arguments must be provided.")

        if create_dir:
            exists = os.path.exists(dirname)
            if exists and not exist_ok:
                raise OSError("Directory {} already exists. Pass exist_ok=True to ignore this error.")
            elif not exists:
                os.makedirs(dirname)

        if require_empty:
            matched = [fname
                       for fname in os.listdir(dirname)
                       if fname.startswith(self._fname_prefix)]

            if len(matched) > 0:
                raise ValueError("Files prefixed with {} are already present "
                                 "in the directory {}. If you want to use this "
                                 "directory anyway, pass `require_empty=False`. "
                                 "".format(filename_prefix, dirname))

    def _save(self, obj, path):
        if not self._atomic:
            torch.save(obj, path)

        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self._dirname)

            try:
                torch.save(obj, tmp.file)

            except BaseException:
                tmp.close()
                os.remove(tmp.name)
                raise

            else:
                tmp.close()
                os.rename(tmp.name, path)

    def __call__(self, engine, to_save):
        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._iteration += 1

        if self._score_function is not None:
            priority = self._score_function(engine)

        else:
            priority = self._iteration
            if (self._iteration % self._save_interval) != 0:
                return

        if (len(self._saved) < self._n_saved) or (self._saved[0][0] < priority):
            saved_objs = []

            for name, obj in to_save.items():
                fname = '{}_{}_{}.pth'.format(self._fname_prefix, name, self._iteration)
                path = os.path.join(self._dirname, fname)

                self._save(obj=obj, path=path)
                saved_objs.append(path)

            self._saved.append((priority, saved_objs))
            self._saved.sort(key=lambda item: item[0])

        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)
