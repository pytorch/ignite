import pytest

from ignite.metrics.nlp import Bleu, RougeL, RougeN


@pytest.mark.parametrize("metric", [Bleu(ngram=1), RougeN(ngram=1), RougeL()])
@pytest.mark.parametrize(
    "candidates,references,expected",
    [
        ([["match"]], [[["match"]], [["extra"]]], "1 != 2"),
        ([["match"], ["extra"]], [[["match"]]], "2 != 1"),
    ],
)
def test_update_rejects_unequal_batch_lengths(metric, candidates, references, expected):
    with pytest.raises(ValueError, match=rf"nb of candidates should be equal to nb of reference lists \({expected}\)"):
        metric.update((candidates, references))
