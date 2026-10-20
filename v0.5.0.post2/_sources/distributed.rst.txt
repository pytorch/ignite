ignite.distributed
==================

Helper module to use distributed settings for multiple backends:

- backends from native torch distributed configuration: "nccl", "gloo", "mpi"

- XLA on TPUs via `pytorch/xla <https://github.com/pytorch/xla>`_

- using `Horovod framework <https://horovod.readthedocs.io/en/stable/>`_ as a backend


Distributed launcher and `auto` helpers
---------------------------------------

We provide a context manager to simplify the code of distributed configuration setup for all above supported backends.
In addition, methods like :meth:`~ignite.distributed.auto.auto_model`, :meth:`~ignite.distributed.auto.auto_optim` and
:meth:`~ignite.distributed.auto.auto_dataloader` helps to adapt in a transparent way provided model, optimizer and data
loaders to existing configuration:

.. code-block:: python

    # main.py

    import ignite.distributed as idist

    def training(local_rank, config, **kwargs):

        print(idist.get_rank(), ": run with config:", config, "- backend=", idist.backend())

        train_loader = idist.auto_dataloader(dataset, batch_size=32, num_workers=12, shuffle=True, **kwargs)
        # batch size, num_workers and sampler are automatically adapted to existing configuration
        # ...
        model = resnet50()
        model = idist.auto_model(model)
        # model is DDP or DP or just itself according to existing configuration
        # ...
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer = idist.auto_optim(optimizer)
        # optimizer is itself, except XLA configuration and overrides `step()` method.
        # User can safely call `optimizer.step()` (behind `xm.optimizer_step(optimizier)` is performed)


    backend = "nccl"  # torch native distributed configuration on multiple GPUs
    # backend = "xla-tpu"  # XLA TPUs distributed configuration
    # backend = None  # no distributed configuration
    # 
    # dist_configs = {'nproc_per_node': 4}  # Use specified distributed configuration if launch as python main.py
    # dist_configs["start_method"] = "fork"  # Add start_method as "fork" if using Jupyter Notebook
    with idist.Parallel(backend=backend, **dist_configs) as parallel:
        parallel.run(training, config, a=1, b=2)

Above code may be executed with `torch.distributed.launch`_ tool or by python and specifying distributed configuration
in the code. For more details, please, see :class:`~ignite.distributed.launcher.Parallel`,
:meth:`~ignite.distributed.auto.auto_model`, :meth:`~ignite.distributed.auto.auto_optim` and
:meth:`~ignite.distributed.auto.auto_dataloader`.

Complete example of CIFAR10 training can be found
`here <https://github.com/pytorch/ignite/tree/master/examples/cifar10>`_.


.. _torch.distributed.launch: https://pytorch.org/docs/stable/distributed.html#launch-utility


ignite.distributed.auto
-----------------------

.. currentmodule:: ignite.distributed.auto

.. autosummary::
    :nosignatures:
    :toctree: generated

    DistributedProxySampler
    auto_dataloader
    auto_model
    auto_optim

.. Note ::

    In distributed configuration, methods :meth:`~ignite.distributed.auto.auto_model`, :meth:`~ignite.distributed.auto.auto_optim`
    and :meth:`~ignite.distributed.auto.auto_dataloader` will have effect only when distributed group is initialized.


ignite.distributed.launcher
---------------------------

.. currentmodule:: ignite.distributed.launcher

.. autosummary::
    :nosignatures:
    :toctree: generated

    Parallel

ignite.distributed.utils
------------------------

This module wraps common methods to fetch information about distributed configuration, initialize/finalize process
group or spawn multiple processes.

.. currentmodule:: ignite.distributed.utils

.. autosummary::
    :nosignatures:
    :autolist:

.. automodule:: ignite.distributed.utils
    :members:

    .. attribute:: has_native_dist_support

        True if `torch.distributed` is available

    .. attribute:: has_xla_support

        True if `torch_xla` package is found
