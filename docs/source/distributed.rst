ignite.distributed
==================

Helper module to use distributed settings for multiple backends:

- backends from native torch distributed configuration: "nccl", "gloo", "mpi"

- XLA on TPUs via `pytorch/xla <https://github.com/pytorch/xla>`_

This module wraps common methods to fetch information about distributed configuration, initialize/finalize process
group or spawn multiple processes.


Distributed launcher context manager
------------------------------------

We provide a context manager to simplify distributed configuration setup for all above supported backends





Other examples:

    - Example to spawn `nprocs` processes that run `fn` with `args`: :meth:`~ignite.distributed.spawn`


ignite.distributed.auto
-----------------------

.. currentmodule:: ignite.distributed.auto

.. automodule:: ignite.distributed.auto
    :members:


ignite.distributed.launcher
---------------------------

.. currentmodule:: ignite.distributed.launcher

.. automodule:: ignite.distributed.launcher
    :members:


ignite.distributed.utils
------------------------

.. currentmodule:: ignite.distributed.utils

.. automodule:: ignite.distributed.utils
    :members:

    .. attribute:: has_native_dist_support

        True if `torch.distributed` is available

    .. attribute:: has_xla_support

        True if `torch_xla` package is found
