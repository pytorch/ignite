Concepts
========

Engine
------

The **essence** of the framework is the class :class:`ignite.engine.Engine`, an abstraction that loops a given number of times over
provided data, executes a processing function and returns a result:

.. code-block:: python

    while epoch < max_epochs:
        # run once on data
        for batch in data:
            output = process_function(batch)

Thus, a model trainer is simply an engine that loops multiple times over the training dataset and updates model parameters.
Similarly, model evaluation can be done with an engine that runs a single time over the validation dataset and computes metrics.
For example, model trainer for a supervised task:

.. code-block:: python

    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.data[0]

    trainer = Engine(update_model)
    trainer.run(data, max_epochs=100)


Events and Handlers
-------------------

To improve the :class:`ignite.engine.Engine`'s flexibility, an event system is introduced that facilitates interaction on each step of
the run:

- *engine is started/completed*
- *epoch is started/completed*
- *batch iteration is started/completed*

Complete list of events can be found at :class:`ignite.engine.Events`.

Thus, user can execute a custom code as an event handler. Let us consider in more detail what happens when
:meth:`ignite.engine.Engine.run` is called:

.. code-block:: python

    fire_event(Events.STARTED)
    while epoch < max_epochs:
        fire_event(Events.EPOCH_STARTED)
        # run once on data
        for batch in data:
            fire_event(Events.ITERATION_STARTED)

            output = process_function(batch)

            fire_event(Events.ITERATION_COMPLETED)
        fire_event(Events.EPOCH_COMPLETED)
    fire_event(Events.COMPLETED)

At first *engine is started* event is fired and all this event handlers are executed (we will see in the next paragraph
how to add event handlers). Next, `while` loop is started and *epoch is started* event occurs, etc. Every time
an event is "fired", attached handlers are executed.

Attaching an event handler is simple using method :meth:`ignite.engine.Engine.add_event_handler` or
:meth:`ignite.engine.Engine.on` decorator:

.. code-block:: python

    trainer = Engine(update_model)

    trainer.add_event_handler(Events.STARTED, lambda engine: print("Start training"))
    # or
    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Another message of start training")

    # attach handler with args, kwargs
    mydata = [1, 2, 3, 4]

    def on_training_ended(engine, data):
        print("Training is ended. mydata={}".format(data))

    trainer.add_event_handler(Events.STARTED, on_training_ended, mydata)

.. Note ::

   User can also register custom events with :meth:`ignite.engine.Engine.register_events`, attach handlers and fire custom events
   calling :meth:`ignite.engine.Engine.fire_event` in any handler or `process_function`.

   See the source code of :class:`ignite.contrib.engines.create_supervised_tbptt_trainer` for an example of usage of
   custom events.


State
-----
A state is introduced in :class:`ignite.engine.Engine` to store the output of the `process_function`, current epoch,
iteration and other helpful information. For example, in case of supervised trainer, we can log computed loss value,
completed iterations and epochs:

.. code-block:: python

    trainer = Engine(update_model)

    def on_iteration_completed(engine):
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        loss = engine.state.output
        print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, iteration, loss))

    trainer.add_event_handler(Events.ITERATION_COMPLETED, on_iteration_completed)

.. Note ::

   A good practice is to use :class:`ignite.engine.State` also as a storage of user data created in update or handler functions.
   For example, we would like to save `new_attribute` in the `state`:

   .. code-block:: python

      def user_handler_function(engine):
          engine.state.new_attribute = 12345


Custom events
-------------
By default, the :class:`ignite.engine.Engine` defines events to interact with a
double loop over epochs and batch around ``process_function``, as defined in the
`Events and Handlers`_ section. Although this setting is sufficient for most
cases such as a suppervised learning setting, some more advanced cases may
require further control over events.

For that purpose, the user can define more events and manually fire them. Let's
go through the process with an example. Let's say you are working on
optimization of neural network and you need to make additional operations
between ``loss.backward()`` (which computes the gradients) and
``optimizer.step()`` (which make an optimization step on the parameters of the
network) of the ``update_model`` function defined in the `Engine`_ section.
There are at least three options to do this:

1. The most straitforward is to write a new ``update_model`` for every of your
   cases;
2. To have more code reuse, you could write a function, that takes a callback
   function and returns an ``update_model`` function that calls that the
   callback function between the two operations;
3. Last, the `update_model` fires an event between the two operations and the
   user (you) latter on decides what is executed.

Option 1. is clearly not good enough as it will duplicate a lot of code. Option
2. let you reuse code but is more convoluted: it requires writting a function
that takes a function and returns a function. Option 3. gets all the modularity
of option 2. while integrating seamlesly in Ignite logic. It can also let you
add multiple ``handler``s for the events you defined.

To fire custom events you need to register them with the
:class:`ignite.engine.Engine` using
:meth:`ignite.engine.Engine.register_events`, then you can use
:meth:`ignite.engine.Engine.fire_event` in your code to trigger that event.

.. code-block:: python

    OPT_EVENT = "opt_event"

    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        trainer.fire_event(OPT_EVENT)
        optimizer.step()
        return loss.data[0]

    trainer = Engine(update_model)
    trainer.register_events(OPT_EVENT)

    # Use like another Event.
    @trainer.on(OPT_EVENT)
    def do_something(engine):
        # Access to the engine and other varaibles in your
        # scope to do what you want.
        pass

    trainer.run(data, max_epochs=100)

If the event ``OPT_EVENT`` was not register, the ``trainer`` would raise an
error at runtime when executing the line ``trainer.fire_event(OPT_EVENT)``.
Ideally, events should be an ``int`` or a ``string``.

When using custom events, a good practice is to use a factory function, such as
:func:`ignite.engine.create_supervised_trainer` to create the engine. That way,
the factory function can register the custom events and it avoids forgetting
them.

.. Note ::
    Events defined by Ignite are already properly registered and fired by the
    :class:`ignite.engine.Engine`. The user need not interact with them except
    for registering handlers.
