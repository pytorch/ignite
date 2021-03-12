import os

import pytest
import torch
from rouge_score import rouge_scorer, tokenize

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics import Rouge
from ignite.metrics.rouge import _lcs, _ngramify, _safe_divide


def test_wrong_inputs():
    with pytest.raises(TypeError):
        rouge = Rouge(beta="l", metric="rouge-1")

    with pytest.raises(ValueError):
        rouge = Rouge(metric="m")

    with pytest.raises(ValueError):
        rouge = Rouge(metric="rouge-0")

    with pytest.raises(ValueError):
        rouge = Rouge(metric="rouge-1", aggregate="test")

    rouge = Rouge()
    with pytest.raises(NotComputableError):
        rouge.compute()


@pytest.mark.parametrize(
    "metric, aggregate, y_indices, y_pred_indices, expected",
    [
        ("rouge-1", "single", [8, 3, 2], [], 0.0),
        ("rouge-1", "single", [], [8, 3, 2], 0.0),
        ("rouge-1", "single", [8, 3, 2], [8], 0.5),
        ("rouge-2", "single", [8, 3, 2], [8, 3], 2 / 3),
        ("rouge-L", "single", [8, 3, 2], [8, 2], 0.8),
    ],
)
def test_rouge(metric, aggregate, y_indices, y_pred_indices, expected):

    rouge = Rouge(metric=metric, beta=1.0, aggregate=aggregate)

    y = ["a" * i for i in y_indices]
    y_pred = ["a" * i for i in y_pred_indices]

    rouge.update((y_pred, y))

    assert rouge.compute() == expected


@pytest.mark.parametrize(
    "metric, aggregate, y_indices, y_pred_indices, expected",
    [
        ("rouge-L", "mean", [[8, 2], [], [], []], [8, 3, 2], 0.2),
        ("rouge-L", "max", [[8, 2], [], [], []], [8, 3, 2], 0.8),
    ],
)
def test_rouge_multi(metric, aggregate, y_indices, y_pred_indices, expected):

    rouge = Rouge(metric=metric, beta=1.0, aggregate=aggregate)

    y = [["a" * i for i in index] for index in y_indices]
    y_pred = ["a" * i for i in y_pred_indices]

    rouge.update((y_pred, y))

    assert rouge.compute() == expected


def test_lcs():
    ref = [1, 2, 3, 4, 5]
    cl = [2, 5, 3, 4]

    assert _lcs(ref, cl) == 3


def test_ngramify():
    y = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i in range(1, 9):
        assert len(_ngramify(y, i)) == len(y) - i + 1


def test_safe_divide():
    assert _safe_divide(1.0, 0.0) == 0
    assert _safe_divide(0.0, 1.0) == 0
    assert _safe_divide(0.0, 0.0) == 0
    assert _safe_divide(1.0, 2.0) == 0.5


def test_rouge_against_rouge155():
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

    rouge1 = Rouge(metric="rouge-1", beta=1.0)
    rouge1.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rouge1.compute() == scores["rouge1"].fmeasure

    rouge2 = Rouge(metric="rouge-2", beta=1.0)
    rouge2.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rouge2.compute() == scores["rouge2"].fmeasure

    rougel = Rouge(metric="rouge-L", beta=1.0)
    rougel.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])

    assert rougel.compute() == scores["rougeL"].fmeasure


def test_compute():
    def _test_compute(rouge1, rougel, scorer):
        acc_rouge1 = 0
        acc_rougel = 0
        y_pred_list = ["the cat was found under the bed", "the tiny little cat was found under the big funny bed"]
        y_list = ["the cat was under the bed", "the cat was under the bed"]
        for i, (y_pred, y) in enumerate(zip(y_pred_list, y_list), start=1):
            score = scorer.score(y_pred, y)
            acc_rouge1 += score["rouge1"].fmeasure
            acc_rougel += score["rougeL"].fmeasure
            rouge1.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])
            rougel.update([tokenize.tokenize(y_pred, None), tokenize.tokenize(y, None)])
            assert isinstance(rouge1.compute(), torch.Tensor)
            assert isinstance(rougel.compute(), torch.Tensor)
            assert rouge1.compute() == acc_rouge1 / i
            assert rougel.compute() == acc_rougel / i

    _test_compute(
        Rouge(metric="rouge-1", beta=1.0),
        Rouge(metric="rouge-l", beta=1.0),
        rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False),
    )


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
        m = Rouge(metric="rouge-1", beta=1.0, device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        acc_score = 0
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        for i in range(n_iters):
            acc_score += scorer.score(y_preds[i], y[i])["rouge1"].fmeasure
        assert m.compute() == acc_score / n_iters
        assert "rouge" in engine.state.metrics

    def _test_l(metric_device):
        engine = Engine(update)
        m = Rouge(metric="rouge-L", beta=1.0, device=metric_device)
        m.attach(engine, "rouge")

        data = list(range(n_iters))
        engine.run(data=data, max_epochs=1)

        acc_score = 0
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        for i in range(n_iters):
            acc_score += scorer.score(y_preds[i], y[i])["rougeL"].fmeasure
        assert m.compute() == acc_score / n_iters
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
        rouge = Rouge(metric="rouge-1", beta=1.0, device=metric_device)
        dev = rouge._device
        assert dev == metric_device, f"{dev} vs {metric_device}"

        y_pred = "the tiny little cat was found under the big funny bed"
        y = "the cat was under the bed"
        rouge.update([y_pred.split(), y.split()])
        dev = rouge._rougetotal.device
        assert dev == metric_device, f"{dev} vs {metric_device}"


def test_accumulator_detached():
    rouge = Rouge(metric="rouge-1", beta=1.0)

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
