import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import WordErrorRate


def test_wer_wrong_inputs():
    wer = WordErrorRate()

    with pytest.raises(NotComputableError, match=r"Error rate must have at least one valid reference sequence"):
        wer.compute()

    with pytest.raises(ValueError, match=r"y_pred and y must have the same length"):
        wer.update((["a", "b"], ["a"]))

    with pytest.raises(ValueError, match=r"y_pred and y must have the same length"):
        wer.update((["a"], ["a", "b"]))


def test_wer_compute():
    wer = WordErrorRate()

    # Exact match
    wer.update((["hello world", "test sequence"], ["hello world", "test sequence"]))
    assert pytest.approx(wer.compute()) == 0.0

    # 1 Substitution
    wer.reset()
    wer.update((["hello word"], ["hello world"]))
    # 1 error / 2 words = 0.5
    assert pytest.approx(wer.compute()) == 0.5

    # 1 Deletion
    wer.reset()
    wer.update((["hello"], ["hello world"]))
    # 1 error / 2 words = 0.5
    assert pytest.approx(wer.compute()) == 0.5

    # 1 Insertion
    wer.reset()
    wer.update((["hello world test"], ["hello world"]))
    # 1 error / 2 words = 0.5
    assert pytest.approx(wer.compute()) == 0.5

    # Completely different
    wer.reset()
    wer.update((["completely different string"], ["hello world test sequence"]))
    # 'completely', 'different', 'string' vs 'hello', 'world', 'test', 'sequence'
    # 4 references. 3 predicted. It will be 4 errors (3 substitutions, 1 deletion).
    assert pytest.approx(wer.compute()) == 1.0


def test_wer_batching():
    wer = WordErrorRate()
    # Batch 1
    wer.update((["the cat sat", "hello world"], ["the bat sat", "hello"]))
    # Batch 2
    wer.update((["test string"], ["test string again"]))

    # 1 sub (the bat sat) = 1_e / 3_ref
    # 1 ins (hello world) = 1_e / 1_ref
    # 1 del (test string again) = 1_e / 3_ref
    # Total errors = 3
    # Total refs = 3 + 1 + 3 = 7
    assert pytest.approx(wer.compute()) == 3 / 7
