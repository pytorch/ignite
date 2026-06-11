Quick start
===========

Welcome to **PyTorch-Ignite** quick start guide that just covers the essentials of getting a project up and walking through the code.

In several lines of this given code, you can get your model trained and validated as shown below:

Code
----

.. code-block:: python

    from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
    from ignite.metrics import Accuracy, Loss

    model = Net()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    criterion = nn.NLLLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    trainer.run(train_loader, max_epochs=100)


**Note**: Complete code can be found in the file `examples/mnist/mnist.py <https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist.py>`_.

Explanation
-----------

Now let's break up the code and review it in details. In the first 4 lines, we define our model, training and validation
datasets (as `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_), optimizer and loss function:

.. code-block:: python

    model = Net()
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    criterion = nn.NLLLoss()

Next we define trainer and evaluator engines. In this example, we are using helper methods as
:meth:`~ignite.engine.create_supervised_trainer` and :meth:`~ignite.engine.create_supervised_evaluator`:

.. code-block:: python

    trainer = create_supervised_trainer(model, optimizer, criterion)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

Objects ``trainer`` and ``evaluator`` are instances of :class:`~ignite.engine.engine.Engine` - main component of Ignite.
:class:`~ignite.engine.engine.Engine` is an abstraction over your training/validation loop.


In general, we can define ``trainer`` and ``evaluator`` using directly :class:`~ignite.engine.engine.Engine` class and
custom training/validation step logic:

.. code-block:: python

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            return y_pred, y

    evaluator = Engine(validation_step)


Note that the helper function :meth:`~ignite.engine.create_supervised_evaluator` is to create an evaluator which accepts an
argument ``metrics``:

.. code-block:: python

    metrics={
        'accuracy': Accuracy(),
        'nll': Loss(loss)
    }

Here we define two metrics: *accuracy* and *loss* to compute on validation dataset. More information on
metrics can be found at :doc:`metrics`.


The most interesting part of the code snippet is adding event handlers. :class:`~ignite.engine.engine.Engine` allows to add handlers on
various events that triggers during the run. When an event is triggered, attached handlers (functions) are executed. Thus, for
logging purposes we add a function to be executed at the end of every ``log_interval``-th iteration:

.. code-block:: python

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

or equivalently without the decorator

.. code-block:: python

    def log_training_loss(engine):
        print(f"Epoch[{engine.state.epoch}] Loss: {engine.state.output:.2f}")

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

When an epoch ends, we want to compute training and validation metrics [#f1]_. For that purpose, we can run previously defined
``evaluator`` on ``train_loader`` and ``val_loader``. Therefore we attach two additional handlers to the trainer on epoch
complete event:

.. code-block:: python

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")


.. Note ::

   Function :meth:`~ignite.engine.engine.Engine.add_event_handler` (as well as :meth:`~ignite.engine.engine.Engine.on` decorator)
   also accepts optional `args`, `kwargs` to be passed to the handler. For example:

   .. code-block:: python

      trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss, train_loader)


Finally, we start the engine on the training dataset and run it during 100 epochs:

.. code-block:: python

    trainer.run(train_loader, max_epochs=100)


**Where to go next?** To understand better the concepts of the library, please read :doc:`concepts`.


.. rubric:: Footnotes

.. [#f1] In this example, we follow a pattern that requires a second pass through the training set. This
   could be expensive on large datasets (even taking a subset). Another more common pattern is to accumulate
   measures online over an epoch in the training loop. In this case, metrics are aggregated on a moving model,
   and thus, we do not want to encourage this pattern. However, if a user still likes to implement the
   last pattern, it can be easily done by attaching metrics to the trainer as following:

   .. code-block:: python

        def custom_output_transform(x, y, y_pred, loss):
            return {
                "y": y,
                "y_pred": y_pred,
                "loss": loss.item()
            }

        trainer = create_supervised_trainer(
            model, optimizer, criterion, device, output_transform=custom_output_transform
        )

        # Attach metrics:
        val_metrics = {
            "accuracy": Accuracy(),
            "nll": Loss(criterion)
        }
        for name, metric in val_metrics.items():
            metric.attach(trainer, name)
