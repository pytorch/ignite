ignite.distributed
==================

Helper module to use distributed settings for multiple backends:

- backends from native torch distributed configuration: "nccl", "gloo", "mpi"

- XLA on TPUs via `pytorch/xla <https://github.com/pytorch/xla>`_

This module wraps common methods to fetch information about distributed configuration, initialize/finalize process
group or spawn multiple processes.


Examples:

    - Example to spawn `nprocs` processes that run `fn` with `args`: :meth:`~ignite.distributed.spawn`


.. currentmodule:: ignite.distributed

.. automodule:: ignite.distributed
    :members:
    :imported-members:

    .. attribute:: has_xla_support

        True if `torch_xla` package is found
