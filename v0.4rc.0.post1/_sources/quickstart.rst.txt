Quickstart
==========

Welcome to Ignite quickstart guide that just gives essentials of getting a project up and running.

In several lines you can get your model training and validating:

Code
----

.. code-block:: python

    from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
    from ignite.metrics import Accuracy, Loss

    model = Net()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    loss = torch.nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(),
                                                'nll': Loss(loss)
                                                })

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    trainer.run(train_loader, max_epochs=100)


Complete code can be found in the file `examples/mnist/mnist.py <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist.py>`_.

Explanation
-----------

Now let's break up the code and review it in details. In the first 4 lines we define our model, training and validation
datasets (as `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_), optimizer and loss function:

.. code-block:: python

    model = Net()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    loss = torch.nn.NLLLoss()

Next we define trainer and evaluator engines. The main component of Ignite is the :class:`~ignite.engine.engine.Engine`, an abstraction over your
training loop. Getting started with the engine is easy, the constructor only requires one things:

- `update_function`: a function that receives the engine and a batch and have a role to update your model.

In the above example we are using helper methods :meth:`~ignite.engine.create_supervised_trainer` and :meth:`~ignite.engine.create_supervised_evaluator`:

.. code-block:: python

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': Accuracy(),
                                                'nll': Loss(loss)
                                                })

However, we could also define trainer and evaluator using :class:`~ignite.engine.engine.Engine`. If we take a look into the source code of
:meth:`~ignite.engine.create_supervised_trainer` and :meth:`~ignite.engine.create_supervised_evaluator`, we can observe the following pattern:

.. code-block:: python

    def create_engine(*args, **kwargs):

        def _update(engine, batch):
            # Update function logic
            pass

        return Engine(_update)

And update functions of the trainer and evaluator are simply:

.. code-block:: python

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device)
            y_pred = model(x)
            return y_pred, y

Note that the helper function :meth:`~ignite.engine.create_supervised_evaluator` to create an evaluator accepts an
argument `metrics`:

.. code-block:: python

    metrics={
        'accuracy': Accuracy(),
        'nll': Loss(loss)
    }

where we define two metrics: *accuracy* and *loss* to compute on validation dataset. More information on
metrics can be found at :doc:`metrics`.


The most interesting part of the code snippet is adding event handlers. :class:`~ignite.engine.engine.Engine` allows to add handlers on
various events that fired during the run. When an event is fired, attached handlers (functions) are executed. Thus, for
logging purposes we added a function to be executed after every iteration:

.. code-block:: python

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        print("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))

or equivalently without the decorator

.. code-block:: python

    def log_training_loss(engine):
        print("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

When an epoch ends we want compute training and validation metrics [#f1]_. For that purpose we can run previously defined
`evaluator` on `train_loader` and `val_loader`. Therefore we attach two additional handlers to the trainer on epoch
complete event:

.. code-block:: python

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))


.. Note ::

   Function :meth:`~ignite.engine.engine.Engine.add_event_handler` (as well as :meth:`~ignite.engine.engine.Engine.on` decorator) also accepts optional `args`, `kwargs` to be passed
   to the handler. For example:

   .. code-block:: python

      trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss, train_loader)


Finally, we start the engine on the training dataset and run it during 100 epochs:

.. code-block:: python

    trainer.run(train_loader, max_epochs=100)


.. rubric:: Footnotes

.. [#f1]

   In this example we follow a pattern that requires a second pass through the training set. This
   could be expensive on large datasets (even taking a subset). Another more common pattern is to accumulate
   measures online over an epoch in the training loop. In this case metrics are aggregated on a moving model,
   and thus, we do not want to encourage this pattern. However, if user still would like to implement the
   last pattern, it can be easily done redefining trainer's update function and attaching metrics as following:

   .. code-block:: python

       def create_supervised_trainer(model, optimizer, loss_fn, metrics={}, device=None):

           def _update(engine, batch):
               model.train()
               optimizer.zero_grad()
               x, y = _prepare_batch(batch, device=device)
               y_pred = model(x)
               loss = loss_fn(y_pred, y)
               loss.backward()
               optimizer.step()
               return loss.item(), y_pred, y

           def _metrics_transform(output):
               return output[1], output[2]

           engine = Engine(_update)

           for name, metric in metrics.items():
               metric._output_transform = _metrics_transform
               metric.attach(engine, name)

           return engine

