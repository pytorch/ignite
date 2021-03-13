ignite.handlers
===============

.. contents:: Complete list of handlers
    :local:

Checkpoint
~~~~~~~~~~
.. autoclass:: ignite.handlers.checkpoint.Checkpoint
    :members: reset, setup_filename_pattern, load_objects, state_dict, load_state_dict, get_default_score_fn

BaseSaveHandler
~~~~~~~~~~~~~~~
.. autoclass:: ignite.handlers.checkpoint.BaseSaveHandler
    :members: __call__, remove

DiskSaver
~~~~~~~~~
.. autoclass:: ignite.handlers.checkpoint.DiskSaver

ModelCheckpoint
~~~~~~~~~~~~~~~
.. autoclass:: ignite.handlers.checkpoint.ModelCheckpoint

EarlyStopping
~~~~~~~~~~~~~
.. autoclass:: ignite.handlers.early_stopping.EarlyStopping

Timer
~~~~~
.. autoclass:: ignite.handlers.timing.Timer
    :members:

TerminateOnNan
~~~~~~~~~~~~~~
.. autoclass:: ignite.handlers.terminate_on_nan.TerminateOnNan

global_step_from_engine
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ignite.handlers.global_step_from_engine

TimeLimit
~~~~~~~~~
.. autoclass:: ignite.handlers.time_limit.TimeLimit
   :members:
