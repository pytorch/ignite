FAQ
===

In this section we grouped answers on frequently asked questions and some best practices of using `ignite`.


Each engine has its own Events
------------------------------

It is important to understand that engines have their own events. For example, we defined a trainer and an evaluator:

.. code-block:: python

    @trainer.on(Events.EPOCH_COMPLETED)
    def in_training_loop_on_epoch_completed(engine):
        evaluator.run(val_loader) # this starts another loop on validation dataset to compute metrics

    @evaluator.on(Events.COMPLETED)
    def when_validation_loop_is_done(engine):
        # do something with computed metrics etc
        # -> early stopping or reduce LR on plateau
        # or just log them


Trainer engine has its own loop and runs multiple times over the training dataset. When a training epoch is over we
launch evaluator engine and run a single time of over the validation dataset. **Evaluator has its own loop**. Therefore,
it runs only one epoch and :attr:`~ignite.engine.Events.EPOCH_COMPLETED` is equivalent to :attr:`~ignite.engine.Events.COMPLETED`.
As a consequence, the following code is correct too:

.. code-block:: python

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    best_model_saver = ModelCheckpoint('/tmp/models', 'best', score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {'mymodel': model})


More details `Events and Handlers <concepts.html#events-and-handlers>`_.


Creating Custom Events based on Forward/Backward Pass
-----------------------------------------------------

There are cases where the user might want to add events based on the loss calculation and backward pass. Ignite provides
flexibility to the user to allow for this:

.. code-block:: python

    class BackpropEvents(Enum):
        """
        Events based on back propagation
        """
        BACKWARD_STARTED = 'backward_started'
        BACKWARD_COMPLETED = 'backward_completed'
        OPTIM_STEP_COMPLETED = 'optim_step_completed'

    def update(engine, batch):
        model.train()
        opitmizer.zero_grad()
        x, y = process_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        engine.fire_event(BackpropEvents.BACKWARD_STARTED)
        loss.backward()
        engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)
        optimizer.step()
        engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
        
        return loss.item()

    trainer = Engine(update)
    trainer.register_events(*BackpropEvents)

    @trainer.on(BackpropEvents.BACKWARD_STARTED)
    def function_before_backprop(engine):
        # insert custom function here


More detailed implementation can be found in `TBPTT Trainer <_modules/ignite/contrib/engines/tbptt.html#create_supervised_tbptt_trainer>`_.


Creating Custom Events based on Iteration and Epoch
---------------------------------------------------

Another type of custom event could be based on number of iteration and epochs. Ignite has :attr:`~ignite.contrib.handlers.custom_events.CustomPeriodicEvent`, which allows the user to
define events based on number of elapsed iterations/epochs.


Gradients accumulation
----------------------

A best practice to use if we need to increase effectively the batchsize on limited GPU resources. There several ways to
do this, the most simple is the following:

.. code-block:: python

    accumulation_steps = 4

    def update_fn(engine, batch):
        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss.item()

    trainer = Engine(update_fn)

Based on `this blog article <https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255>`_ and
`this code <https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py>`_.


Other answers can be found on the github among the issues labeled by
`question <https://github.com/pytorch/ignite/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aquestion+>`_.
