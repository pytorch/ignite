import os
import warnings

import pytest
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import Bleu

from . import CorpusForTest

corpus = CorpusForTest(lower_split=True)


def test_wrong_inputs():

    with pytest.raises(ValueError, match=r"ngram order must be greater than zero"):
        Bleu(ngram=0)

    with pytest.raises(ValueError, match=r"Smooth is not valid"):
        Bleu(smooth="fake")

    with pytest.raises(ValueError, match=r"nb of candidates should be equal to nb of reference lists"):
        Bleu()._corpus_bleu(references=[[0], [0]], candidates=[[0]])

    with pytest.raises(NotComputableError):
        Bleu().compute()


@pytest.mark.parametrize(
    "candidate, references",
    [
        ([["a"], ["a"]]),
        ([["a", "a", "a", "b", "c"]], [[["a", "b", "c"], ["a", "a", "d"]]]),
        corpus.sample_1,
        corpus.sample_2,
        corpus.sample_3,
        corpus.sample_4,
    ],
)
def test_corpus_bleu(candidate, references):
    print(candidate, references)
    for i in range(1, 8):
        weights = tuple([1 / i] * i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = corpus_bleu(references, candidate, weights=weights)
        bleu = Bleu(ngram=i)
        assert pytest.approx(reference) == bleu._corpus_bleu(references, candidate)
        bleu.update((candidate[0], references[0]))
        assert pytest.approx(reference) == bleu.compute()


@pytest.mark.parametrize(
    "candidate, references",
    [
        ([["a", "a", "a", "b", "c"]], [[["a", "b", "c"], ["a", "a", "d"]]]),
        corpus.sample_1,
        corpus.sample_2,
        corpus.sample_3,
        corpus.sample_4,
    ],
)
def test_corpus_bleu_smooth1(candidate, references):
    for i in range(1, 8):
        weights = tuple([1 / i] * i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = corpus_bleu(
                references, candidate, weights=weights, smoothing_function=SmoothingFunction().method1
            )
        bleu = Bleu(ngram=i, smooth="smooth1")
        assert reference == bleu._corpus_bleu(references, candidate)
        bleu.update((candidate[0], references[0]))
        assert reference == bleu.compute()


@pytest.mark.parametrize(
    "candidate, references",
    [
        ([["a", "a", "a", "b", "c"]], [[["a", "b", "c"], ["a", "a", "d"]]]),
        corpus.sample_1,
        corpus.sample_2,
        corpus.sample_3,
        corpus.sample_4,
    ],
)
def test_corpus_bleu_nltk_smooth2(candidate, references):
    for i in range(1, 8):
        weights = tuple([1 / i] * i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = corpus_bleu(
                references, candidate, weights=weights, smoothing_function=SmoothingFunction().method2
            )
        bleu = Bleu(ngram=i, smooth="nltk_smooth2")
        assert reference == bleu._corpus_bleu(references, candidate)
        bleu.update((candidate[0], references[0]))
        assert reference == bleu.compute()


@pytest.mark.parametrize(
    "candidate, references",
    [
        ([["a", "a", "a", "b", "c"]], [[["a", "b", "c"], ["a", "a", "d"]]]),
        corpus.sample_1,
        corpus.sample_2,
        corpus.sample_3,
        corpus.sample_4,
    ],
)
def test_corpus_bleu_smooth2(candidate, references):
    for i in range(1, 3):
        weights = tuple([1 / i] * i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = corpus_bleu(
                references, candidate, weights=weights, smoothing_function=SmoothingFunction().method2
            )
        bleu = Bleu(ngram=i, smooth="smooth2")
        assert reference == bleu._corpus_bleu(references, candidate)
        bleu.update((candidate[0], references[0]))
        assert reference == bleu.compute()


def test_bleu():
    bleu = Bleu(ngram=4, smooth="smooth2")
    bleu.update((corpus.cand_1, corpus.references_1))
    bleu.update((corpus.cand_2a, corpus.references_2))
    bleu.update((corpus.cand_2b, corpus.references_2))
    bleu.update((corpus.cand_3, corpus.references_2))
    value = bleu._corpus_bleu([corpus.references_1], [corpus.cand_1])
    value += bleu._corpus_bleu([corpus.references_2], [corpus.cand_2a])
    value += bleu._corpus_bleu([corpus.references_2], [corpus.cand_2b])
    value += bleu._corpus_bleu([corpus.references_2], [corpus.cand_3])
    assert bleu.compute() == value / 4


def test_bleu_batch():

    # Batch size 1
    refs = [corpus.references_2]
    hypotheses = [corpus.cand_2a]
    bleu = Bleu(ngram=4)
    bleu.update((hypotheses, refs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_bleu_score = (sentence_bleu(refs[0], hypotheses[0]) + sentence_bleu(refs[0], hypotheses[0])) / 2
    assert bleu.compute() == reference_bleu_score

    # Batch size 3
    hypotheses = [corpus.cand_1, corpus.cand_2a, corpus.cand_2b]
    refs = [corpus.references_1, corpus.references_2, corpus.references_2]
    bleu.reset()
    bleu.update((hypotheses, refs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_bleu_score = (
            sentence_bleu(refs[0], hypotheses[0])
            + sentence_bleu(refs[1], hypotheses[1])
            + sentence_bleu(refs[2], hypotheses[2])
        ) / 3
    assert bleu.compute() == reference_bleu_score


def _test_distrib_integration(device):

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
        m = Bleu(ngram=4, smooth="smooth2")
        m.attach(engine, "bleu")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "bleu" in engine.state.metrics

        ref_bleu = 0
        for candidate, references in data:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref_bleu += corpus_bleu(
                    [references],
                    [candidate],
                    weights=[0.25, 0.25, 0.25, 0.25],
                    smoothing_function=SmoothingFunction().method2,
                )

        assert pytest.approx(engine.state.metrics["bleu"]) == ref_bleu / len(data)

    _test("cpu")

    if device.type != "xla":
        _test(idist.device())


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Skip if no GPU")
def test_distrib_nccl_gpu(distributed_context_single_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_distrib_gloo_cpu_or_gpu(distributed_context_single_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_distrib_integration, (device,), np=nproc, do_init=True)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_gloo_cpu_or_gpu(distributed_context_multi_node_gloo):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.multinode_distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
@pytest.mark.skipif("GPU_MULTINODE_DISTRIB" not in os.environ, reason="Skip if not multi-node distributed")
def test_multinode_distrib_nccl_gpu(distributed_context_multi_node_nccl):

    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_distrib_integration(device)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_distrib_integration(device)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
