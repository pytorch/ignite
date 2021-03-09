import os
from math import isclose

import pytest
import torch
from rouge_score import rouge_scorer, tokenize

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Rouge
from ignite.metrics.rouge import _lcs


def test_zero_div():
    rouge = Rouge()
    with pytest.raises(NotComputableError):
        rouge.compute()


def test_input():
    with pytest.raises(TypeError):
        rouge = Rouge(beta="l", n=1)

    with pytest.raises(ValueError):
        rouge = Rouge(n="m")

    with pytest.raises(ValueError):
        rouge = Rouge(n=-1)


def testrougeempty():
    rouge = Rouge(n=1, beta=1.0)
    y = "testing one two"
    y_pred = ""
    rouge.update([y_pred.split(), y.split()])
    assert rouge.compute() == 0.0


def testrouge1():
    rouge = Rouge(n=1, beta=1.0)
    y = "testing one two"
    y_pred = "testing"
    rouge.update([y_pred.split(), y.split()])
    assert rouge.compute() == 0.5


def testrouge2():
    rouge = Rouge(n=2, beta=1.0)
    y = "testing one two"
    y_pred = "testing one"
    rouge.update([y_pred.split(), y.split()])
    assert rouge.compute() == 2 / 3


def testrougel():
    rouge = Rouge(n="l", beta=1.0)
    y = "testing one two"
    y_pred = "testing two"
    rouge.update([y_pred.split(), y.split()])
    assert rouge.compute() == 0.8


def testlcs():
    ref = [1, 2, 3, 4, 5]
    cl = [2, 5, 3, 4]

    assert _lcs(ref, cl) == 3


def testrougeagainstrouge155():
    y_pred = """rFDJCRtion Ht-LM EKtDXkME,yz'RBr q'wer wrojNbN wL,b .a-'XdQggyFl jB-RPP'iyOIcUxi n cw-WeFyu vC MoBL Xdn g
    wkvcEiGvKtion BDFhrpMer pstion sbKao Q m qier LMmed HqqLFXe,XPY,J XsurkMeo ,ed nB'wH'bWVHjWFEer
    tQ.saefZwJtKrTlixYpMMNJtion UCAPwNHeYVjD"""
    y = """ZfbCUIUuePaiLVUlCaUXxkpu XPeWing tUHfPMuZ',-Xd Y BrUgVJing M-HV.-DgdDaY.rFDJCRing Ht-LM EKBDXkME,yz'RBr
    q'wtion wIojNbN wL,b .a-'XdQggyFl jB - RPP'iyOIcUxer tKM L KsJdPByEtAor fE-Qg Dpdbring cw-WeFyu vC MoBL Xdn g
    wkvcEiGvKtion BDFhrpMtion psing sbKao Q m qiing LMmer HqqLFXe,XPY,J XsurkMer ,ed nB'wH'bWVHjWFEing
    tQ.saefZwJtKrTlixYpMMsJing UCAPwNHeYVjDing c T BUySKtion gMPfJpwGw p'NvxAoping eu pBwMBKV'I DNxqelhu,PHPxDEq,mtion
    SN"""

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(y_pred, y)

    rouge1 = Rouge(n=1, beta=1.0)
    rouge1.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rouge1.compute() == scores["rouge1"].fmeasure

    rouge2 = Rouge(n=2, beta=1.0)
    rouge2.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rouge2.compute() == scores["rouge2"].fmeasure

    rougel = Rouge(n="l", beta=1.0)
    rougel.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rougel.compute() == scores["rougeL"].fmeasure


def test_compute():
    rouge = Rouge()

    y_pred = "the cat was found under the bed"
    y = "the cat was under the bed"
    rouge.update([y_pred.split(), y.split()])
    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.8571428571428571

    y_pred = "the tiny little cat was found under the big funny bed"
    y = "the cat was under the bed"
    rouge.update([y_pred.split(), y.split()])
    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.7012987012987013

    rouge = Rouge(n="l")

    y_pred = "the cat was found under the bed"
    y = "the cat was not under the bed"
    rouge.update([y_pred.split(), y.split()])

    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.8571428571428571

    y_pred = "the cat was found under the big funny bed"
    y = "the tiny little cat was under the bed"
    rouge.update([y_pred.split(), y.split()])

    assert isinstance(rouge.compute(), torch.Tensor)
    assert rouge.compute() == 0.7619047619047619


def _test_distrib_integration(device):
    import ignite.distributed as idist
    from ignite.engine import Engine
    from ignite.metrics import Rouge

    n_iters = 2

    y = ["Hi", "Hello"]
    y_preds = ["Hi there", "Hello How are you"]

    def update(engine, i):
        return (y[i].split(), y_preds[i].split())

    def _test_n(metric_device):
        engine = Engine(update)
        m = Rouge(device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rouge" in engine.state.metrics

    def _test_l(metric_device):
        engine = Engine(update)
        m = Rouge(n="l", device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        assert "rouge" in engine.state.metrics

    _test_n("cpu")
    _test_l("cpu")
    if device.type != "xla":
        _test_n(idist.device())
        _test_l(idist.device())


def _test_distrib_accumulator_device(device):

    metric_devices = [torch.device("cpu")]
    if torch.device(device).type != "xla":
        metric_devices.append(idist.device())

    for metric_device in metric_devices:
        rouge = Rouge(device=metric_device)
        dev = rouge._device
        assert dev == metric_device, f"{dev} vs {metric_device}"

        y_pred = "the tiny little cat was found under the big funny bed"
        y = "the cat was under the bed"
        rouge.update([y_pred.split(), y.split()])
        dev = rouge._rougetotal.device
        assert dev == metric_device, f"{dev} vs {metric_device}"


def test_accumulator_detached():
    rouge = Rouge()

    y_pred = "the cat was found under the big funny bed"
    y = "the tiny little cat was under the bed"
    rouge.update([y_pred.split(), y.split()])

    assert not rouge._rougetotal.requires_grad


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_gpu(local_rank, distributed_context_single_node_nccl):
    device = torch.device(f"cuda:{local_rank}")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_cpu(distributed_context_single_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_distrib_accumulator_device, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_cpu(distributed_context_multi_node_gloo):
    device = torch.device("cpu")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gpu(distributed_context_multi_node_nccl):
    device = torch.device(f"cuda:{distributed_context_multi_node_nccl['local_rank']}")
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)
    _test_distrib_accumulator_device(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
