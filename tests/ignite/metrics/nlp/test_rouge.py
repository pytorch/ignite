import os

import filelock

import nltk
import pytest
import rouge as pyrouge
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import Rouge
from ignite.metrics.nlp.rouge import compute_ngram_scores, RougeL, RougeN

from . import CorpusForTest


@pytest.fixture(scope="session", autouse=True)
def download_nltk_punkt(worker_id, tmp_path_factory):
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    while True:
        try:
            with filelock.FileLock(root_tmp_dir / "nltk_download.lock", timeout=0.2) as fn:
                fn.acquire()
                nltk.download("punkt")
                fn.release()
                break
        except filelock._error.Timeout:
            pass


corpus = CorpusForTest()


@pytest.mark.parametrize(
    "candidate, reference, n, expected_precision, expected_recall",
    [
        ([], [], 1, 0, 0),
        ("abc", "ab", 1, 2 / 3, 2 / 2),
        ("abc", "ab", 2, 1 / 2, 1 / 1),
        ("abc", "ab", 3, 0, 0),
        ("ab", "abc", 1, 2 / 2, 2 / 3),
        ("ab", "cde", 1, 0 / 2, 0 / 3),
        ("aab", "aace", 1, 2 / 3, 2 / 4),
        ("aa", "aaa", 1, 2 / 2, 2 / 3),
        ("aaa", "aa", 1, 2 / 3, 2 / 2),
    ],
)
def test_compute_ngram_scores(candidate, reference, n, expected_precision, expected_recall):
    scores = compute_ngram_scores(candidate, reference, n=n)
    assert pytest.approx(scores.precision()) == expected_precision
    assert pytest.approx(scores.recall()) == expected_recall


def test_wrong_inputs():
    with pytest.raises(ValueError, match=r"ngram order must be greater than zero"):
        RougeN(ngram=0)

    with pytest.raises(ValueError, match=r"alpha must be in interval \[0, 1\]"):
        RougeN(alpha=-1)

    with pytest.raises(ValueError, match=r"alpha must be in interval \[0, 1\]"):
        RougeN(alpha=2)

    with pytest.raises(ValueError, match=r"multiref : valid values are \['best', 'average'\] "):
        RougeN(multiref="")

    with pytest.raises(ValueError, match=r"variant must be 'L' or integer greater to zero"):
        Rouge(variants=["error"])

    with pytest.raises(NotComputableError):
        RougeL().compute()

    with pytest.raises(ValueError):
        Rouge(multiref="unknown")


@pytest.mark.parametrize(
    "ngram, candidate, reference, expected",
    [
        (1, [1, 2, 3], [1, 2], (2 / 3, 2 / 2)),
        (2, [1, 2, 3], [1, 2], (1 / 2, 1 / 1)),
        (1, "abcdef", "zbdfz", (3 / 6, 3 / 5)),
        (2, "abcdef", "zbdfz", (0, 0)),
    ],
)
def test_rouge_n_alpha(ngram, candidate, reference, expected, available_device):
    for alpha in [0, 1, 0.3, 0.5, 0.8]:
        rouge = RougeN(ngram=ngram, alpha=alpha, device=available_device)
        rouge.update(([candidate], [[reference]]))
        results = rouge.compute()
        assert results[f"Rouge-{ngram}-P"] == expected[0]
        assert results[f"Rouge-{ngram}-R"] == expected[1]
        try:
            F = expected[0] * expected[1] / ((1 - alpha) * expected[0] + alpha * expected[1])
        except ZeroDivisionError:
            F = 0
        assert results[f"Rouge-{ngram}-F"] == F


@pytest.mark.parametrize(
    "candidates, references", [corpus.sample_1, corpus.sample_2, corpus.sample_3, corpus.sample_4, corpus.sample_5]
)
def test_rouge_metrics(candidates, references, available_device):
    for multiref in ["average", "best"]:
        # PERL 1.5.5 reference
        apply_avg = multiref == "average"
        apply_best = multiref == "best"
        evaluator = pyrouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=4,
            apply_avg=apply_avg,
            apply_best=apply_best,
            alpha=0.5,
            stemming=False,
            ensure_compatibility=False,
        )
        scores = evaluator.get_scores(candidates, references)

        lower_split_references = [
            [ref.lower().split() for ref in refs_per_candidate] for refs_per_candidate in references
        ]

        lower_split_candidates = [candidate.lower().split() for candidate in candidates]

        m = Rouge(variants=[1, 2, 4, "L"], multiref=multiref, alpha=0.5, device=available_device)
        assert m._device == torch.device(available_device)
        m.update((lower_split_candidates, lower_split_references))
        results = m.compute()

        for key in ["1", "2", "4", "L"]:
            assert pytest.approx(results[f"Rouge-{key}-R"], abs=1e-4) == scores[f"rouge-{key.lower()}"]["r"]
            assert pytest.approx(results[f"Rouge-{key}-P"], abs=1e-4) == scores[f"rouge-{key.lower()}"]["p"]
            assert pytest.approx(results[f"Rouge-{key}-F"], abs=1e-4) == scores[f"rouge-{key.lower()}"]["f"]


def _test_distrib_integration(device):
    from ignite.engine import Engine

    rank = idist.get_rank()

    size = len(corpus.chunks)

    data = []
    for c in corpus.chunks:
        data += idist.get_world_size() * [c]

    def update(_, i):
        candidate, references = data[i + size * rank]
        lower_split_references = [reference.lower().split() for reference in references[0]]
        lower_split_candidate = candidate[0].lower().split()
        return [lower_split_candidate], [lower_split_references]

    def _test(metric_device):
        engine = Engine(update)
        m = Rouge(variants=[1, 2, "L"], alpha=0.5, device=metric_device)
        m.attach(engine, "rouge")

        engine.run(data=list(range(size)), max_epochs=1)

        assert "rouge" in engine.state.metrics

        evaluator = pyrouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=4,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,
            stemming=False,
            ensure_compatibility=False,
        )
        rouge_1_f, rouge_2_f, rouge_l_f = (0, 0, 0)
        for candidate, references in data:
            scores = evaluator.get_scores(candidate, references)
            rouge_1_f += scores["rouge-1"]["f"]
            rouge_2_f += scores["rouge-2"]["f"]
            rouge_l_f += scores["rouge-l"]["f"]
        assert pytest.approx(engine.state.metrics["Rouge-1-F"], abs=1e-4) == rouge_1_f / len(data)
        assert pytest.approx(engine.state.metrics["Rouge-2-F"], abs=1e-4) == rouge_2_f / len(data)
        assert pytest.approx(engine.state.metrics["Rouge-L-F"], abs=1e-4) == rouge_l_f / len(data)

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
