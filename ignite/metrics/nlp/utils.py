from collections import Counter
from typing import Any, Sequence, Tuple

__all__ = ["ngrams", "lcs", "modified_precision"]


def ngrams(sequence: Sequence[Any], n: int) -> Counter:
    """
    Generate the ngrams from a sequence of items

    Args:
        sequence: sequence of items
        n: n-gram order

    Returns:
        A counter of ngram objects

    .. versionadded:: 0.4.5
    """
    return Counter([tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1)])


def lcs(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """
    Compute the length of the longest common subsequence in two sequence of items
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem

    Args:
        seq_a: first sequence of items
        seq_b: second sequence of items

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    m = len(seq_a)
    n = len(seq_b)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def modified_precision(references: Sequence[Sequence[Any]], candidate: Any, n: int) -> Tuple[int, int]:
    """
    Compute the modified precision

    .. math::
       p_{n} = \frac{m_{n}}{l_{n}}

    where m_{n} is the number of matched n-grams between translation T and its reference R, and l_{n} is the
    total number of n-grams in the translation T.

    More details can be found in `Papineni et al. 2002`__.

    __ https://aclanthology.org/P02-1040

    Args:
        references: list of references R
        candidate: translation T
        n: n-gram order

    Returns:
        The length of the longest common subsequence

    .. versionadded:: 0.4.5
    """
    # ngrams of the candidate
    counts = ngrams(candidate, n)

    # union of ngrams of references
    max_counts: Counter = Counter()
    for reference in references:
        max_counts |= ngrams(reference, n)

    # clipped count of the candidate and references
    clipped_counts = counts & max_counts

    return sum(clipped_counts.values()), sum(counts.values())
