import os
import pytest

import ignite.distributed as idist
from ignite.distributed.auto import auto_model, auto_optim, auto_dataloader


def _test_auto_optim():
    pass


@pytest.mark.distributed
def test_auto_optim_gloo(distributed_context_single_node_gloo):
    _test_auto_optim()


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_auto_optim_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_auto_optim, args=(), nprocs=n)
