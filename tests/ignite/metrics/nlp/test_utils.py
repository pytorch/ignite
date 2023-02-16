import pytest

from ignite.metrics.nlp.utils import lcs, modified_precision, ngrams


@pytest.mark.parametrize(
    "sequence, n, expected_keys, expected_values",
    [
        ([], 1, [], []),
        ([0, 1, 2], 1, [(0,), (1,), (2,)], [1, 1, 1]),
        ([0, 1, 2], 2, [(0, 1), (1, 2)], [1, 1]),
        ([0, 1, 2], 3, [(0, 1, 2)], [1]),
        ([0, 0, 0], 1, [(0,)], [3]),
        ([0, 0, 0], 2, [(0, 0)], [2]),
        ("abcde", 4, [("a", "b", "c", "d"), ("b", "c", "d", "e")], [1, 1]),
    ],
)
def test_ngrams(sequence, n, expected_keys, expected_values):
    ngrams_counter = ngrams(sequence=sequence, n=n)
    assert list(ngrams_counter.values()) == expected_values
    assert list(ngrams_counter.keys()) == expected_keys


@pytest.mark.parametrize(
    "seq_a, seq_b, expected",
    [([], [], 0), ([0, 1, 2], [0, 1, 2], 3), ([0, 1, 2], [0, 3, 2], 2), ("academy", "abracadabra", 4)],
)
def test_lcs(seq_a, seq_b, expected):
    assert lcs(seq_a, seq_b) == expected


def test_modified_precision_empty():
    for k in range(1, 5):
        n, d = modified_precision([[]], [], k)
        assert n == 0 and d == 0
        n, d = modified_precision([[]], [0], k)
        assert n == 0 and d == (k == 1)
        n, d = modified_precision([[0]], [], k)
        assert n == 0 and d == 0
        n, d = modified_precision([[]], list(range(k)), k)
        assert n == 0 and d == 1
        n, d = modified_precision([list(range(k))], [], k)
        assert n == 0 and d == 0


@pytest.mark.parametrize(
    "references, candidate, expected",
    [
        ([[0, 0, 0], [1, 2]], [1, 2, 3, 4], ((2, 4), (1, 3), (0, 2))),
        ([[0, 1, 2], [0, 0, 3]], [0, 0, 0, 1, 2], ((4, 5), (3, 4), (1, 3))),
        ([[0, 1, 2], [3, 0, 3]], [3, 0, 0, 1, 2], ((4, 5), (3, 4), (1, 3))),
    ],
)
def test_modified_precision(references, candidate, expected):
    for n, (e_n, e_d) in enumerate(expected, start=1):
        n, d = modified_precision(references, candidate, n)
        assert n == e_n and d == e_d
