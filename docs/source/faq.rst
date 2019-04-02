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


Gradients accumulation
----------------------

A best practice to use if we need to increase effectively the batchsize on limited GPU resources. There several ways to
do this, the most simple is the following:

.. code-block:: python

    accumulation_steps = 4

    def update_fn(engine, batch):
        model.train()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y) / accumulation_steps
        loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()

        return loss.item()

    trainer = Engine(update_fn)

Based on `this blog article <https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255>`_ and
`this code <https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py>`_.


Other answers can be found on the github among the issues labeled by
`question <https://github.com/pytorch/ignite/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aquestion+>`_.
