Ignite
======

.. image:: https://travis-ci.org/pytorch/ignite.svg?branch=master
    :target: https://travis-ci.org/pytorch/ignite

Ignite is a high-level library to help with training neural networks in PyTorch.

*Note: Ignite is currently in alpha, and as such the code is changing rapidly in master. We hope to stabalise the API as soon as possible and keep the examples up to date.*

- `Documentation`_
- `Installation`_
- `Getting Started`_
    - `The Trainer`_
    - `Training & Validation History`_
    - `Events & Event Handlers`_
    - `Logging`_
    - `Metrics`_
- `Examples`_
- `How does this compare to Torchnet?`_
- `Contributing`_

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

    optimizer = ...
    model = ...
    criterion = ...
    def training_update_function(batch):
        model.train()
        optimizer.zero_grad()
        x, y = Variable(batch[0]), Variable(batch[1])
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        return loss.data[0]

You can then construct your :code:`Trainer` and train for `num_epochs` as follows:

.. code-block:: python

    from ignite.trainer import Trainer 
    
    trainer = Trainer(train_dataloader, training_update_function)
    trainer.run(max_epochs=5)

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
- EXCEPTION_RAISED (see `Note on EXCEPTION_RAISED`_ below)

Users can attach multiple handlers to each of these events, which allows them to control aspects of training such as 
early stopping, or reducing the learning rate as well as things such as logging or updating external dashboards like
`Visdom <https://github.com/facebookresearch/visdom>`_ or `TensorBoard <https://www.tensorflow
.org/get_started/summaries_and_tensorboard>`_ (See `Examples`_ for more details on using Visdom).

Event handlers are any callable where the first argument is an instance of the :code:`Trainer`. Users can also pass any other arguments or keyword arguments to their event handlers. For example, if we want to terminate training after 100 iterations if the learning rate hasn't decreased in the last 10 iterations, we could define the following event handler and attach it to the :code:`TRAINING_ITERATION_COMPLETED` event.

.. code-block:: python

    from ignite.trainer import TrainingEvents

    def early_stopping_handler(trainer, min_iterations, lookback=1):
        if trainer.current_iterations >= min_iterations:
            last_loss = trainer.training_history[-1]
            if not any(x < last_loss for x in trainer.training_history[-lookback:]):
                trainer.terminate()

    min_iterations = 100
    trainer.add_event_handler(TrainingEvents.TRAINING_ITERATION_COMPLETED,
                              early_stopping_handler,
                              min_iterations,
                              lookback=5)

Note on EXCEPTION_RAISED
^^^^^^^^^^^^^^^^^^^^^^^^
By default :code:`Ignite` will re-raise any exception caught during training. If you would like to change this
behaviour, you have to register an handler for the :code:`EXCEPTION_RAISED` event. This handler will be called with
the :code:`Engine` object and the caught :code:`Exception` as two first arguments, followed by any :code:`*args` and
:code:`**kwargs` passed to :code:`add_event_handler` call. If this handler is registered, :code:`Ignite` will not
re-raise the exception.

The following example shows how to register a handler which will save the model before re-rasing the exception.

.. code-block:: python

    from ignite.engine import Events

    def save_and_raise(engine, exception, model, path):
        torch.save(model, path)
        raise exception

    trainer.add_event_handler(Events.EXCEPTION_RAISED,
                              save_and_raise,
                              my_model,
                              '/tmp/my_model.pth')

Logging
+++++++
Ignite uses `python's standard library logging module <https://docs.python.org/2/library/logging.html>`_, which means you can integrate the Ignite logs directly into your application logs. To do this, simply attach a log handler to the `ignite` logger:

.. code-block:: python

    import logging
    logger = logging.getLogger('ignite')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

Metrics
+++++++
Ignite supports certain metrics which can be used to classify the performance of a given model. The metrics currently available in :code:`ignite` are:

- :code:`binary_accuracy` : This takes a :code:`History` object (either :code:`training_history` or :code:`validation_history`) and an optional callable transform and computes the binary accuracy which is 1 if the values are equal or 0 otherwise. This is generally used for binary classification tasks
- :code:`categorical_accuracy` : This is the :code:`binary_accuracy` equivalent for multi-class classification where number of classes are greater than 2.
- :code:`top_k_categorical_accuracy` : This computes the Top K classification accuracy, which is a popular mode of evaluating models on larger datasets with higher number of classes. The semantics are similar to :code:`categorical_accuracy` except there is an additional argument for the value of :code:`k`
- :code:`mean_squared_error` : Generally used in regression tasks, this computes the sum of squared deviations between the predicted value and the actual value for a given input datapoint. This function takes a :code:`History` object and an optional callable transform and computes the mean squared error. The square root of this gives the root mean squared error (RMSE).
- :code:`mean_absolute_error` : This is similar to the :code:`mean_squared_error` function, but instead computes the sum of absolute deviations between the predicted value and the actual value for a given input datapoint.

Examples
++++++++
At present, there is an example of how to use ignite to train a digit classifier on MNIST in `examples/
<https://github.com/pytorch/ignite/tree/master/examples>`_, this example covers the following things:

- Attaching custom handlers to training events
- Attaching ignite's handlers to training events
- Using handlers to plot to a visdom server to visualize training loss and validation accuracy

How does this compare to Torchnet?
==================================
Ignite, in spirit is very similar to `torchnet <https://github.com/pytorch/tnt>`_ (and was inspired by torchnet). 

The main differences with torchnet is the level of abstraction for the user. Ignite's higher level of abstraction assumes less about the type of network (or networks) that you are training, and we require the user to define the closure to be run in the training and validation loop. In contrast to this, torchnet creates this closure internally based on the network and optimizer you pass to it. This higher level of abstraction allows for a great deal more of flexibility, such as co-training multiple models (i.e. GANs) and computing/tracking multiple losses and metrics in your training loop.

Ignite also allows for multiple handlers to be attached to events, and a finer granularity of events in the loop.

That being said, there are some things from torchnet we really like and would like to port over, such as the integration with Visdom (and possibly add integration with TensorBoard).

As always, PRs are welcome :)

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Please see the [contribution guidelines](https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md) for more information.
