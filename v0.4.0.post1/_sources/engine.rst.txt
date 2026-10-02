ignite.engine
==============

Main module of the library containing:

- :class:`~ignite.engine.engine.Engine` - abstraction that loops provided data, executes a processing function and returns a result
- :class:`~ignite.engine.events.Events` - events triggered by the :class:`~ignite.engine.engine.Engine` during execution
- :class:`~ignite.engine.events.State` - object to pass internal and user-defined data between event handlers

and helper methods:

- :meth:`~ignite.engine.create_supervised_trainer` - creates single model/optimizer/criterion supervised trainer
- :meth:`~ignite.engine.create_supervised_evaluator` - creates single model supervised evaluation engine


More details about those structures can be found in :doc:`concepts`.


.. currentmodule:: ignite.engine.engine

.. autoclass:: Engine
   :members:

.. autofunction:: ignite.engine.create_supervised_trainer

.. autofunction:: ignite.engine.create_supervised_evaluator


Resuming the training
---------------------

It is possible to resume the training from a checkpoint and approximately reproduce original run's behaviour.
Using Ignite, this can be easily done using :class:`~ignite.handlers.Checkpoint` handler. Engine provides two methods
to serialize and deserialize its internal state :meth:`~ignite.engine.engine.Engine.state_dict` and
:meth:`~ignite.engine.engine.Engine.load_state_dict`. In addition to serializing model, optimizer, lr scheduler etc user can
store the trainer and then resume the training. For example:

.. code-block:: python

    from ignite.engine import Engine, Events
    from ignite.handlers import Checkpoint, DiskSaver

    trainer = ...
    model = ...
    optimizer = ...
    lr_scheduler = ...
    data_loader = ...

    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    handler = Checkpoint(to_save, DiskSaver('/tmp/training', create_dir=True))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
    trainer.run(data_loader, max_epochs=100)

.. code-block:: bash

    ls /tmp/training
    > "checkpoint_50000.pt"

We can then restore the training from the last checkpoint.

.. code-block:: python

    from ignite.handlers import Checkpoint

    trainer = ...
    model = ...
    optimizer = ...
    lr_scheduler = ...
    data_loader = ...

    to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    checkpoint = torch.load(checkpoint_file)
    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

    trainer.run(train_loader, max_epochs=100)


It is also possible to store checkpoints every N iterations and continue the training from one of these checkpoints, i.e
from iteration.

Complete examples that resumes the training from a checkpoint can be found here:

- `save/resume MNIST <https://github.com/pytorch/ignite/tree/master/examples/mnist#training-save--resume>`_
- `save/resume Distributed CIFAR10 <https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10#check-resume-training>`_


ignite.engine.events
--------------------

.. currentmodule:: ignite.engine.events

.. autoclass:: Events
   :members:

.. autoclass:: State

.. autoclass:: RemovableEventHandle
   :members:
   :undoc-members:


ignite.engine.deterministic
---------------------------

Deterministic training
``````````````````````

In general, it is rather difficult task to achieve deterministic and reproducible trainings as it relies on multiple
aspects, e.g. data version, code version, software environment, hardware etc. According to `PyTorch documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_:
there are some steps to take in order to make computations deterministic on your specific problem on one specific
platform and PyTorch release:

- setup random state seed

- set `cudnn to deterministic <https://pytorch.org/docs/stable/notes/randomness.html#cudnn>`_ if applicable

By default, these two options can be enough to run and rerun experiments in a deterministic way. Ignite's engine does not impact this behaviour.


In this module we provide helper methods and classes to make additional ":ref:`Dataflow synchronization`"
to ensure that model sees the same data for a given epoch:

- :class:`~ignite.engine.deterministic.DeterministicEngine`
- :class:`~ignite.engine.deterministic.ReproducibleBatchSampler`


.. currentmodule:: ignite.engine.deterministic

.. automodule:: ignite.engine.deterministic
   :members:


Dataflow synchronization
------------------------

Ignite provides an option to control the dataflow by synchronizing random state on epochs. In this way, for a given
iteration/epoch the dataflow can be the same for a given seed. More precisely it is roughly looks like:

.. code-block:: python

    for e in range(num_epochs):
        set_seed(seed + e)
        do_single_epoch_iterations(dataloader)


In addition, if data provider is ``torch.utils.data.DataLoader``, batch data indices can be made completely deterministic.
Here is a trivial example of usage:

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from ignite.engine import DeterministicEngine, Events
    from ignite.utils import manual_seed


    def random_train_data_loader(size):
        data = torch.arange(0, size)
        return DataLoader(data, batch_size=4, shuffle=True)


    def print_train_data(engine, batch):
        i = engine.state.iteration
        e = engine.state.epoch
        print("train", e, i, batch.tolist())

    trainer = DeterministicEngine(print_train_data)

    print("Original Run")
    manual_seed(56)
    trainer.run(random_train_data_loader(40), max_epochs=2, epoch_length=5)

    print("Resumed Run")
    # Resume from 2nd epoch
    trainer.load_state_dict({"epoch": 1, "epoch_length": 5, "max_epochs": 2, "rng_states": None})
    manual_seed(56)
    trainer.run(random_train_data_loader(40))

.. code-block:: text

    Original Run
    train 1 1 [31, 13, 3, 4]
    train 1 2 [23, 18, 6, 16]
    train 1 3 [10, 8, 33, 36]
    train 1 4 [1, 37, 19, 9]
    train 1 5 [20, 30, 14, 26]
    train 2 6 [29, 35, 38, 34]
    train 2 7 [7, 22, 12, 17]
    train 2 8 [25, 21, 24, 15]
    train 2 9 [39, 5, 2, 28]
    train 2 10 [27, 11, 32, 0]
    Resumed Run
    train 2 6 [29, 35, 38, 34]
    train 2 7 [7, 22, 12, 17]
    train 2 8 [25, 21, 24, 15]
    train 2 9 [39, 5, 2, 28]
    train 2 10 [27, 11, 32, 0]


We can see that the data samples are exactly the same between original and resumed runs.

Complete examples that simulates a crash on a defined iteration and resumes the training from a checkpoint can be found
here:

- `save/resume MNIST <https://github.com/pytorch/ignite/tree/master/examples/mnist#training-save--resume>`_
- `save/resume Distributed CIFAR10 <https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10#check-resume-training>`_


.. Note ::

    In case when input data is `torch.utils.data.DataLoader`, previous batches are skipped and the first provided batch
    corresponds to the batch after the checkpoint iteration. Internally, while resuming, previous datapoint indices are just
    skipped without fetching the data.

.. warning::

    However, while resuming from iteration, random data augmentations are not synchronized in the middle of the epoch and
    thus batches remaining until the end of the epoch can be different of those from the initial run.

.. warning::

    However, please, keep in mind that there can be an issue with dataflow synchronization on every epoch
    if user's handler synchronizes the random state, for example, by calling periodically ``torch.manual_seed(seed)`` during
    the run. This can have an impact on the dataflow:

    .. code-block:: python

        def random_train_data_generator():
            while True:
                yield torch.randint(0, 100, size=(1, ))

        trainer = DeterministicEngine(print_train_data)

        @trainer.on(Events.ITERATION_COMPLETED(every=3))
        def user_handler():
            # handler synchronizes the random state
            torch.manual_seed(12)
            a = torch.rand(1)

        trainer.run(random_train_data_generator(), max_epochs=3, epoch_length=5);

    .. code-block:: text

        train 1 1 [32]
        train 1 2 [29]
        train 1 3 [40]
        train 1 4 [3]  <---
        train 1 5 [22]
        train 2 6 [77]
        train 2 7 [3]  <---
        train 2 8 [22]
        train 2 9 [77]
        train 2 10 [3] <---
        train 3 11 [22]
        train 3 12 [77]
        train 3 13 [3] <---
        train 3 14 [22]
        train 3 15 [77]

    Initially, the function ``random_train_data_generator()`` generates randomly data batches using the random state set
    up by ``trainer``. This is intended behaviour until ``user_handler()`` is called.
    After ``user_handler()`` execution, random state is altered and thus ``random_train_data_generator()`` will produce
    random batches based on altered random state.

    We provide helper decorator :meth:`~ignite.engine.deterministic.keep_random_state` to save and restore random states for
    `torch`, `numpy` and `random`. Therefore, we can deal with described issue using this decorator:

    .. code-block:: python

        from ignite.engine.deterministic import keep_random_state

        @trainer.on(Events.ITERATION_COMPLETED(every=3))
        @keep_random_state
        def user_handler():
            # handler synchronizes the random state
            torch.manual_seed(12)
            a = torch.rand(1)

