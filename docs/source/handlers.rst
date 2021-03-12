ignite.handlers
===============

Complete list of handlers
-------------------------

.. currentmodule:: ignite.handlers

.. autosummary::
    :nosignatures:
    :autolist:

Checkpoint
~~~~~~~~~~
.. autoclass:: Checkpoint
    :members: reset, setup_filename_pattern, load_objects, state_dict, load_state_dict, get_default_score_fn

BaseSaveHandler
~~~~~~~~~~~~~~~
.. autoclass:: ignite.handlers.checkpoint.BaseSaveHandler
    :members: __call__, remove

DiskSaver
~~~~~~~~~
.. autoclass:: DiskSaver

ModelCheckpoint
~~~~~~~~~~~~~~~
.. autoclass:: ModelCheckpoint

EarlyStopping
~~~~~~~~~~~~~
.. autoclass:: EarlyStopping

Timer
~~~~~
.. autoclass:: Timer
    :members:

TerminateOnNan
~~~~~~~~~~~~~~
.. autoclass:: TerminateOnNan

global_step_from_engine
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: global_step_from_engine

TimeLimit
~~~~~~~~~
.. autoclass:: TimeLimit
   :members:
