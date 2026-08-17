ignite.contrib.handlers
=======================

Contribution module of handlers


Parameter scheduler [deprecated]
--------------------------------

.. deprecated:: 0.4.4
   Use :class:`~ignite.handlers.param_scheduler.ParamScheduler` instead, will be removed in version 0.6.0.

   Was moved to :ref:`param-scheduler-label`.

LR finder [deprecated]
----------------------

.. deprecated:: 0.4.4
    Use :class:`~ignite.handlers.lr_finder.FastaiLRFinder` instead, will be removed in version 0.6.0.

Time profilers [deprecated]
---------------------------

.. deprecated:: 0.4.6
    Use :class:`~ignite.handlers.time_profilers.BasicTimeProfiler` instead, will be removed in version 0.6.0.
    Use :class:`~ignite.handlers.time_profilers.HandlersTimeProfiler` instead, will be removed in version 0.6.0.

Loggers
-------

.. currentmodule:: ignite.contrib.handlers

.. autosummary::
    :nosignatures:
    :toctree: ../generated
    :recursive:

    base_logger
    clearml_logger
    mlflow_logger
    neptune_logger
    polyaxon_logger
    tensorboard_logger
    tqdm_logger
   
    visdom_logger
    wandb_logger

.. seealso::

    Below are a comprehensive list of examples of various loggers.

    * See `tensorboardX mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_tensorboard_logger.py>`_
      and `CycleGAN and EfficientNet notebooks <https://github.com/pytorch/ignite/tree/master/examples/notebooks>`_ for detailed usage.

    * See `visdom mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_visdom_logger.py>`_ for detailed usage.

    * See `neptune mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_neptune_logger.py>`_ for detailed usage.

    * See `tqdm mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_tqdm_logger.py>`_ for detailed usage.

    * See `wandb mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_wandb_logger.py>`_ for detailed usage.

    * See `clearml mnist example <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_clearml_logger.py>`_ for detailed usage.
