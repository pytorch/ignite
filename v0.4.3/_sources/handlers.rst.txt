ignite.handlers
===============

Complete list of handlers
-------------------------

.. currentmodule:: ignite.handlers

.. autosummary::
    :nosignatures:
    :autolist:

.. autoclass:: Checkpoint
    :members: reset, setup_filename_pattern, load_objects, state_dict, load_state_dict, get_default_score_fn

.. autoclass:: ignite.handlers.checkpoint.BaseSaveHandler
    :members: __call__, remove

.. autoclass:: DiskSaver

.. autoclass:: ModelCheckpoint

.. autoclass:: EarlyStopping

.. autoclass:: Timer
    :members:

.. autoclass:: TerminateOnNan

.. autofunction:: global_step_from_engine

.. autoclass:: TimeLimit
   :members: