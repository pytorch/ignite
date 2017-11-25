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


Examples
++++++++
Coming soon

Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.
