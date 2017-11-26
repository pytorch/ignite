Ignite
======

.. image:: https://travis-ci.org/pytorch/ignite.svg?branch=master
    :target: https://travis-ci.org/pytorch/ignite

Ignite is a high-level library to help with training neural networks in PyTorch.


Documentation
=============
API documentation, examples and tutorials coming soon.


Installation
============

From Source:

.. code-block:: bash

   python setup.py install


Getting Started
===============

The Trainer
+++++++++++
The main component of Ignite is the :code:`Trainer`, an abstraction over your training loop. Getting started with the trainer is easy, the constructor only requires two things:

- :code:`training_data`: A collection of training batches allowing repeated iteration (e.g., list or DataLoader)
- :code:`training_update_function`: A function which is passed a :code:`batch` and passes data through and updates your model

Optionally, you can also provide `validation_data` and `validation_update_function` for evaluating on your validation set.

Given a :code:`model`, :code:`criterion` and :code:`optimizer` your :code:`training_update_function` will be something like:

.. code-block:: python

    optimzer = ...
    model = ...
    criterion = ...
    def training_update_function(batch):
        model.train()
        optimizer.zero_grad()
        input, target = Variable(batch[0]), Variable(batch[1])
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.data[0]

You can then construct your :code:`Trainer` and train for `num_epochs` as follows:

.. code-block:: python

    trainer = Trainer(train_dataloader, training_update_function)
    trainer.run(num_epochs=5)

Training & Validation History
+++++++++++++++++++++++++++++
The return values of your training and validation update functions are stored in the `Trainer` in the members `training_history` and `validation_history`. These can be accessed via event handlers (see below) and used for updating metrics, logging etc. Importantly, the return type of your update functions need not just be the loss, but can be any type (list, typle, dict, tensors etc.).


Events & Event Handlers
++++++++++++++++++++++++
The :code:`Trainer` emits events during the training loop, which the user can attach event handlers to. The events that are emitted are defined in :code:`ignite.trainer.TrainingEvents`, which at present are:

- EPOCH_STARTED
- EPOCH_COMPLETED
- TRAINING_EPOCH_STARTED
- TRAINING_EPOCH_COMPLETED
- VALIDATION_STARTING
- VALIDATION_COMPLETED
- TRAINING_STARTED
- TRAINING_COMPLETED
- TRAINING_ITERATION_STARTED
- TRAINING_ITERATION_COMPLETED
- VALIDATION_ITERATION_STARTED
- VALIDATION_ITERATION_COMPLETED
- EXCEPTION_RAISED

Users can attach multiple handlers to each of these events, which allows them to control aspects of training such as early stopping, or reducing the learning rate as well as things such as logging or updating external dashboards like `Visdom <https://github.com/facebookresearch/visdom>`_ or `TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`_.

Event handlers are any callable where the first argument is an instance of the :code:`Trainer`. Users can also pass any orther arguments or keywword arguments to their event handlers. For example, if we want to terminate training after 100 iterations if the learning rate hasn't decreased in the last 10 iterations, we could define the following event handler and attach it to the :code:`TRAINING_ITERATION_COMPLETED` event.

.. code-block:: python

    from ignite.trainer import TrainingEvents

    def early_stopping_handler(trainer, min_iterations, lookback=1):
        if trainer.current_iterations >= min_iterations:
            last_loss = trainer.training_history[-1]
            if not any(x < last_loss for x in trainer.training_history[-lookback:-1]):
                trainer.terminate()

    min_iterations = 100
    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_COMPLETED,
                              early_stopping_handler,
                              min_iterations,
                              lookback=5)

Examples
++++++++
Coming soon

Logging
+++++++
Ignite uses `python's standard library logging module <https://docs.python.org/2/library/logging.html>`_, which means you can integrate the Ignite logs directly into your application logs. To do this, simply attach a log handler to the `ignite` logger:

.. code-block:: python

    import logging
    logger = logging.getLogger('ignite')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.
