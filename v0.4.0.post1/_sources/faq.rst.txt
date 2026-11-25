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
it runs only one epoch and `Events.EPOCH_COMPLETED` is equivalent to `Events.COMPLETED`.
As a consequence, the following code is correct too:

.. code-block:: python

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    best_model_saver = ModelCheckpoint('/tmp/models', 'best', score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {'mymodel': model})


More details :ref:`Events and Handlers`.


Creating Custom Events based on Forward/Backward Pass
-----------------------------------------------------

There are cases where the user might want to add events based on the loss calculation and backward pass. Ignite provides
flexibility to the user to allow for this:

.. code-block:: python

    class BackpropEvents(EventEnum):
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


Gradients accumulation
----------------------

A best practice to use if we need to increase effectively the batch size on limited GPU resources. There several ways to
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


Working with iterators
----------------------

If data provider for training or validation is an iterator (infinite or finite with known or unknown size), here are
basic examples of how to setup trainer or evaluator.


Infinite iterator for training
``````````````````````````````

Let's use an infinite data iterator as training dataflow

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    def infinite_iterator(batch_size):
        while True:
            batch = torch.rand(batch_size, 3, 32, 32)
            yield batch

    def train_step(trainer, batch):
        # ...
        s = trainer.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch.norm())
        )

    trainer = Engine(train_step)
    # We need to specify epoch_length to define the epoch
    trainer.run(infinite_iterator(4), epoch_length=5, max_epochs=3)

In this case we will obtain the following output:

.. code-block:: text

    1/3 : 1 - 63.862
    1/3 : 2 - 64.042
    1/3 : 3 - 63.936
    1/3 : 4 - 64.141
    1/3 : 5 - 64.767
    2/3 : 6 - 63.791
    2/3 : 7 - 64.565
    2/3 : 8 - 63.602
    2/3 : 9 - 63.995
    2/3 : 10 - 63.943
    3/3 : 11 - 63.831
    3/3 : 12 - 64.276
    3/3 : 13 - 64.148
    3/3 : 14 - 63.920
    3/3 : 15 - 64.226

If we do not specify `epoch_length`, we can stop the training explicitly by calling :meth:`~ignite.engine.engine.Engine.terminate`
In this case, there will be only a single epoch defined.

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    def infinite_iterator(batch_size):
        while True:
            batch = torch.rand(batch_size, 3, 32, 32)
            yield batch

    def train_step(trainer, batch):
        # ...
        s = trainer.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch.norm())
        )

    trainer = Engine(train_step)

    @trainer.on(Events.ITERATION_COMPLETED(once=15))
    def stop_training():
        trainer.terminate()

    trainer.run(infinite_iterator(4))

We obtain the following output:

.. code-block:: text

    1/1 : 1 - 63.862
    1/1 : 2 - 64.042
    1/1 : 3 - 63.936
    1/1 : 4 - 64.141
    1/1 : 5 - 64.767
    1/1 : 6 - 63.791
    1/1 : 7 - 64.565
    1/1 : 8 - 63.602
    1/1 : 9 - 63.995
    1/1 : 10 - 63.943
    1/1 : 11 - 63.831
    1/1 : 12 - 64.276
    1/1 : 13 - 64.148
    1/1 : 14 - 63.920
    1/1 : 15 - 64.226


Same code can be used for validating models.


Finite iterator with unknown length
```````````````````````````````````

Let's use a finite data iterator but with unknown length (for user). In case of training, we would like to perform
several passes over the dataflow and thus we need to restart the data iterator when it is exhausted.
In the code, we do not specify `epoch_length` which will be automatically determined.

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    def finite_unk_size_data_iter():
        for i in range(11):
            yield i

    def train_step(trainer, batch):
        # ...
        s = trainer.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch)
        )

    trainer = Engine(train_step)

    @trainer.on(Events.DATALOADER_STOP_ITERATION)
    def restart_iter():
        trainer.state.dataloader = finite_unk_size_data_iter()

    data_iter = finite_unk_size_data_iter()
    trainer.run(data_iter, max_epochs=5)


In case of validation, the code is simply

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    def finite_unk_size_data_iter():
        for i in range(11):
            yield i

    def val_step(evaluator, batch):
        # ...
        s = evaluator.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch)
        )

    evaluator = Engine(val_step)

    data_iter = finite_unk_size_data_iter()
    evaluator.run(data_iter)


Finite iterator with known length
`````````````````````````````````

Let's use a finite data iterator with known size for training or validation.
If we need to restart the data iterator, we can do this either as in case of
unknown size by attaching the restart handler on `@trainer.on(Events.DATALOADER_STOP_ITERATION)`,
but here we will do this explicitly on iteration:

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    size = 11

    def finite_size_data_iter(size):
        for i in range(size):
            yield i

    def train_step(trainer, batch):
        # ...
        s = trainer.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch)
        )

    trainer = Engine(train_step)

    @trainer.on(Events.ITERATION_COMPLETED(every=size))
    def restart_iter():
        trainer.state.dataloader = finite_size_data_iter(size)

    data_iter = finite_size_data_iter(size)
    trainer.run(data_iter, max_epochs=5)


In case of validation, the code is simply

.. code-block:: python

    import torch
    from ignite.engine import Engine, Events

    torch.manual_seed(12)

    size = 11

    def finite_size_data_iter(size):
        for i in range(size):
            yield i

    def val_step(evaluator, batch):
        # ...
        s = evaluator.state
        print(
            "{}/{} : {} - {:.3f}".format(s.epoch, s.max_epochs, s.iteration, batch)
        )

    evaluator = Engine(val_step)

    data_iter = finite_size_data_iter(size)
    evaluator.run(data_iter)


Switching data provider during the training
-------------------------------------------

User can easily switch data provider during the training using :meth:`~ignite.engine.engine.Engine.set_data`.
See an example in the documentation of the method.


Time profiling during training
------------------------------

User can fetch times in several manners depending on complexity of required time profiling:

Single epoch and total time
```````````````````````````

Simpliest way to fetch time of single epoch and complete training is to use
``engine.state.times["EPOCH_COMPLETED"]`` and ``engine.state.times["COMPLETED"]``:

.. code-block:: python

    trainer = ...

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_time():
        print("{}: {}".format(trainer.state.epoch, trainer.state.times["EPOCH_COMPLETED"]))

    @trainer.on(Events.COMPLETED)
    def log_total_time():
        print("Total: {}".format(trainer.state.times["COMPLETED"]))


For details, see :class:`~ignite.engine.events.State`.


Detailed profiling
``````````````````

User can setup :class:`~ignite.contrib.handlers.time_profilers.BasicTimeProfiler` to fetch times spent in data
processing, training step, event handlers:

.. code-block:: python

    from ignite.contrib.handlers import BasicTimeProfiler

    trainer = ...

    # Create an object of the profiler and attach an engine to it
    profiler = BasicTimeProfiler()
    profiler.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_intermediate_results():
        profiler.print_results(profiler.get_results())

    trainer.run(dataloader, max_epochs=3)

Typical output:

.. code-block:: text

     ----------------------------------------------------
    | Time profiling stats (in seconds):                 |
     ----------------------------------------------------
    total  |  min/index  |  max/index  |  mean  |  std

    Processing function:
    157.46292 | 0.01452/1501 | 0.26905/0 | 0.07730 | 0.01258

    Dataflow:
    6.11384 | 0.00008/1935 | 0.28461/1551 | 0.00300 | 0.02693

    Event handlers:
    2.82721

    - Events.STARTED: []
    0.00000

    - Events.EPOCH_STARTED: []
    0.00006 | 0.00000/0 | 0.00000/17 | 0.00000 | 0.00000

    - Events.ITERATION_STARTED: ['PiecewiseLinear']
    0.03482 | 0.00001/188 | 0.00018/679 | 0.00002 | 0.00001

    - Events.ITERATION_COMPLETED: ['TerminateOnNan']
    0.20037 | 0.00006/866 | 0.00089/1943 | 0.00010 | 0.00003

    - Events.EPOCH_COMPLETED: ['empty_cuda_cache', 'training.<locals>.log_elapsed_time', ]
    2.57860 | 0.11529/0 | 0.14977/13 | 0.12893 | 0.00790

    - Events.COMPLETED: []
    not yet triggered

For details, see :class:`~ignite.contrib.handlers.time_profilers.BasicTimeProfiler`.


Custom time measures
````````````````````

Custom time measures can be performed using :class:`~ignite.handlers.Timer`. See its docstring for details.


Other questions
---------------

Other answers can be found on the github among the issues labeled by
`question <https://github.com/pytorch/ignite/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3Aquestion+>`_.
