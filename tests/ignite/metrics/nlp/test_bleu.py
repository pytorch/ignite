import os
import warnings
from collections import Counter

import numpy as np
import pytest
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import Bleu

from . import CorpusForTest

corpus = CorpusForTest(lower_split=True)


def to_float32_if_mps(x, device):
    if isinstance(x, torch.Tensor) and device == "mps" and x.dtype == torch.float64:
        return x.to(torch.float32)
    elif isinstance(x, np.ndarray) and device == "mps" and x.dtype == np.float64:
        return x.astype(np.float32)
    return x


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"ngram order must be greater than zero"):
        Bleu(ngram=0)

    with pytest.raises(ValueError, match=r"Smooth is not valid"):
        Bleu(smooth="fake")

    with pytest.raises(ValueError, match=r"nb of candidates should be equal to nb of reference lists"):
        Bleu()._corpus_bleu(references=[[0], [0]], candidates=[[0]])

    with pytest.raises(NotComputableError):
        Bleu().compute()

    with pytest.raises(ValueError, match='Average must be either "macro" or "micro"'):
        Bleu(average="macros")


parametrize_args = (
    "candidates, references",
    [
        ([["a", "a", "a", "b", "c"]], [[["a", "b", "c"], ["a", "a", "d"]]]),
        corpus.sample_1,
        corpus.sample_2,
        corpus.sample_3,
        corpus.sample_4,
    ],
)


def _test(candidates, references, average, smooth="no_smooth", smooth_nltk_fn=None, ngram_range=8, device="cpu"):

    candidates = to_float32_if_mps(candidates, device)
    references = to_float32_if_mps(references, device)

    for i in range(1, ngram_range):
        weights = tuple([1 / i] * i)
        bleu = Bleu(ngram=i, average=average, smooth=smooth, device=device)

        if average == "macro":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference = sentence_bleu(
                    references[0], candidates[0], weights=weights, smoothing_function=smooth_nltk_fn
                )
            computed = bleu._sentence_bleu(references[0], candidates[0])
            if isinstance(computed, torch.Tensor):
                computed = computed.cpu().float().item()
            assert np.allclose(computed, reference, rtol=1e-6)

        elif average == "micro":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference = corpus_bleu(references, candidates, weights=weights, smoothing_function=smooth_nltk_fn)
            computed = bleu._corpus_bleu(references, candidates)
            if isinstance(computed, torch.Tensor):
                computed = computed.cpu().float().item()
            assert np.allclose(computed, reference, rtol=1e-6)

        bleu.update((candidates, references))
        computed = bleu.compute()
        if isinstance(computed, torch.Tensor):
            computed = computed.cpu().float().item()
        assert np.allclose(computed, reference, rtol=1e-6)


@pytest.mark.parametrize(*parametrize_args)
def test_macro_bleu(candidates, references, available_device):
    _test(candidates, references, "macro", device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_micro_bleu(candidates, references, available_device):
    _test(candidates, references, "micro", device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_macro_bleu_smooth1(candidates, references, available_device):
    _test(candidates, references, "macro", "smooth1", SmoothingFunction().method1, device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_micro_bleu_smooth1(candidates, references, available_device):
    _test(candidates, references, "micro", "smooth1", SmoothingFunction().method1, device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_macro_bleu_nltk_smooth2(candidates, references, available_device):
    _test(candidates, references, "macro", "nltk_smooth2", SmoothingFunction().method2, device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_micro_bleu_nltk_smooth2(candidates, references, available_device):
    _test(candidates, references, "micro", "nltk_smooth2", SmoothingFunction().method2, device=available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_macro_bleu_smooth2(candidates, references, available_device):
    _test(candidates, references, "macro", "smooth2", SmoothingFunction().method2, 3, available_device)


@pytest.mark.parametrize(*parametrize_args)
def test_micro_bleu_smooth2(candidates, references, available_device):
    _test(candidates, references, "micro", "smooth2", SmoothingFunction().method2, 3, device=available_device)


def test_accumulation_macro_bleu(available_device):
    bleu = Bleu(ngram=4, smooth="smooth2", device=available_device)
    assert bleu._device == torch.device(available_device)
    cand_1 = to_float32_if_mps(corpus.cand_1, available_device)
    cand_2a = to_float32_if_mps(corpus.cand_2a, available_device)
    cand_2b = to_float32_if_mps(corpus.cand_2b, available_device)
    cand_3 = to_float32_if_mps(corpus.cand_3, available_device)
    ref_1 = to_float32_if_mps(corpus.references_1, available_device)
    ref_2 = to_float32_if_mps(corpus.references_2, available_device)

    bleu.update(([cand_1], [ref_1]))
    bleu.update(([cand_2a], [ref_2]))
    bleu.update(([cand_2b], [ref_2]))
    bleu.update(([cand_3], [ref_2]))
    value = bleu._sentence_bleu(ref_1, cand_1)
    value += bleu._sentence_bleu(ref_2, cand_2a)
    value += bleu._sentence_bleu(ref_2, cand_2b)
    value += bleu._sentence_bleu(ref_2, cand_3)
    computed = bleu.compute()
    if isinstance(computed, torch.Tensor):
        computed = computed.cpu().float().item()
    assert np.allclose(computed, value / 4, rtol=1e-6)


def test_accumulation_micro_bleu(available_device):
    bleu = Bleu(ngram=4, smooth="smooth2", average="micro", device=available_device)
    assert bleu._device == torch.device(available_device)
    cand_1 = to_float32_if_mps(corpus.cand_1, available_device)
    cand_2a = to_float32_if_mps(corpus.cand_2a, available_device)
    cand_2b = to_float32_if_mps(corpus.cand_2b, available_device)
    cand_3 = to_float32_if_mps(corpus.cand_3, available_device)
    ref_1 = to_float32_if_mps(corpus.references_1, available_device)
    ref_2 = to_float32_if_mps(corpus.references_2, available_device)

    bleu.update(([cand_1], [ref_1]))
    bleu.update(([cand_2a], [ref_2]))
    bleu.update(([cand_2b], [ref_2]))
    bleu.update(([cand_3], [ref_2]))
    value = bleu._corpus_bleu(
        [ref_1, ref_2, ref_2, ref_2],
        [cand_1, cand_2a, cand_2b, cand_3],
    )
    assert bleu.compute() == value


def test_bleu_batch_macro(available_device):
    bleu = Bleu(ngram=4, device=available_device)
    assert bleu._device == torch.device(available_device)

    # Batch size 3
    hypotheses = [to_float32_if_mps(c, available_device) for c in [corpus.cand_1, corpus.cand_2a, corpus.cand_2b]]
    refs = [
        to_float32_if_mps(r, available_device) for r in [corpus.references_1, corpus.references_2, corpus.references_2]
    ]
    bleu.update((hypotheses, refs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_bleu_score = (
            sentence_bleu(refs[0], hypotheses[0])
            + sentence_bleu(refs[1], hypotheses[1])
            + sentence_bleu(refs[2], hypotheses[2])
        ) / 3
    reference_bleu_score = np.float32(reference_bleu_score)
    computed = bleu.compute()
    if isinstance(computed, torch.Tensor):
        computed = computed.cpu().float().item()
    assert np.allclose(computed, reference_bleu_score, rtol=1e-6)

    value = 0
    for _hypotheses, _refs in zip(hypotheses, refs):
        value += bleu._sentence_bleu(_refs, _hypotheses)
        bleu.update(([_hypotheses], [_refs]))

    ref_1 = value / len(refs)
    ref_2 = bleu.compute()

    assert np.allclose(ref_1, reference_bleu_score, rtol=1e-6)
    assert np.allclose(ref_2, reference_bleu_score, rtol=1e-6)


def test_bleu_batch_micro(available_device):
    bleu = Bleu(ngram=4, average="micro", device=available_device)
    assert bleu._device == torch.device(available_device)

    # Batch size 3
    hypotheses = [to_float32_if_mps(c, available_device) for c in [corpus.cand_1, corpus.cand_2a, corpus.cand_2b]]
    refs = [
        to_float32_if_mps(r, available_device) for r in [corpus.references_1, corpus.references_2, corpus.references_2]
    ]
    bleu.update((hypotheses, refs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_bleu_score = corpus_bleu(refs, hypotheses)
    assert pytest.approx(bleu.compute()) == reference_bleu_score

    assert pytest.approx(bleu._corpus_bleu(refs, hypotheses)) == reference_bleu_score


@pytest.mark.parametrize(
    "candidates, references",
    [
        (corpus.cand_1, corpus.references_1),
        (corpus.cand_2a, corpus.references_2),
        (corpus.cand_2b, corpus.references_2),
        (corpus.cand_1, corpus.references_1),
    ],
)
def test_n_gram_counter(candidates, references, available_device):
    bleu = Bleu(ngram=4, device=available_device)
    assert bleu._device == torch.device(available_device)

    candidates = to_float32_if_mps(candidates, available_device)
    references = to_float32_if_mps(references, available_device)

    hyp_length, ref_length = bleu._n_gram_counter([references], [candidates], Counter(), Counter())
    hyp_length = int(hyp_length)
    ref_length = int(ref_length)
    assert hyp_length == len(candidates)

    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - len(candidates)), ref_len))

    assert ref_length == closest_ref_len


def _test_macro_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()

    size = len(corpus.chunks)

    data = []
    for c in corpus.chunks:
        data += idist.get_world_size() * [c]

    def update(_, i):
        return data[i + size * rank]

    def _test(device):
        engine = Engine(update)
        m = Bleu(ngram=4, smooth="smooth2", device=device)
        m.attach(engine, "bleu")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "bleu" in engine.state.metrics

        ref_bleu = 0
        for candidates, references in data:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_bleu += sentence_bleu(
                    references[0],
                    candidates[0],
                    weights=[0.25, 0.25, 0.25, 0.25],
                    smoothing_function=SmoothingFunction().method2,
                )

        assert pytest.approx(engine.state.metrics["bleu"]) == ref_bleu / len(data)

    _test("cpu")

    if device.type != "xla":
        _test(idist.device())


def _test_micro_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()

    size = len(corpus.chunks)

    data = []
    for c in corpus.chunks:
        data += idist.get_world_size() * [c]

    def update(_, i):
        return data[i + size * rank]

    def _test(metric_device):
        engine = Engine(update)
        m = Bleu(ngram=4, smooth="smooth2", average="micro", device=metric_device)
        m.attach(engine, "bleu")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "bleu" in engine.state.metrics

        ref_bleu = 0
        references = []
        candidates = []
        for _candidates, _references in data:
            references.append(_references[0])
            candidates.append(_candidates[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_bleu += corpus_bleu(
                references,
                candidates,
                weights=[0.25, 0.25, 0.25, 0.25],
                smoothing_function=SmoothingFunction().method2,
            )

        assert pytest.approx(engine.state.metrics["bleu"]) == ref_bleu

    _test("cpu")

    if device.type != "xla":
        _test(idist.device())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_macro_distrib_integration, (device,), np=nproc, do_init=True)
    gloo_hvd_executor(_test_micro_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_macro_distrib_integration(device)
    _test_micro_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
