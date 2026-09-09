ignite.handlers
===============

Complete list of handlers
-------------------------

    - :class:`~ignite.handlers.Checkpoint`
    - :class:`~ignite.handlers.checkpoint.BaseSaveHandler`
    - :class:`~ignite.handlers.DiskSaver`
    - :class:`~ignite.handlers.ModelCheckpoint`
    - :class:`~ignite.handlers.EarlyStopping`
    - :class:`~ignite.handlers.Timer`
    - :class:`~ignite.handlers.TerminateOnNan`


.. currentmodule:: ignite.handlers

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
