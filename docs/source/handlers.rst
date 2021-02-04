ignite.handlers
===============

Complete list of handlers
-------------------------

.. currentmodule:: ignite.handlers

.. autosummary::
    :nosignatures:
    :autolist:

.. autoclass:: Checkpoint
    :members: load_objects

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