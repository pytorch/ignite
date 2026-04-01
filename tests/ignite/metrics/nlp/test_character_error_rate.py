import pytest
import torch

import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import CharacterErrorRate


def test_cer_wrong_inputs():
    cer = CharacterErrorRate()

    with pytest.raises(NotComputableError, match=r"Error rate must have at least one valid reference sequence"):
        cer.compute()

    with pytest.raises(ValueError, match=r"y_pred and y must have the same length"):
        cer.update((["a", "b"], ["a"]))

    with pytest.raises(ValueError, match=r"y_pred and y must have the same length"):
        cer.update((["a"], ["a", "b"]))


def test_cer_compute():
    cer = CharacterErrorRate()

    # Exact match
    cer.update((["hello", "world"], ["hello", "world"]))
    assert pytest.approx(cer.compute()) == 0.0

    # 1 Substitution
    cer.reset()
    cer.update((["heldo"], ["hello"]))
    # 1 error / 5 chars = 0.2
    assert pytest.approx(cer.compute()) == 0.2

    # 1 Deletion
    cer.reset()
    cer.update((["helo"], ["hello"]))
    # 1 error / 5 chars = 0.2
    assert pytest.approx(cer.compute()) == 0.2

    # 1 Insertion
    cer.reset()
    cer.update((["helllo"], ["hello"]))
    # 1 error / 5 chars = 0.2
    assert pytest.approx(cer.compute()) == 0.2

    # Completely different
    cer.reset()
    cer.update((["a"], ["bcd"]))
    # 3 errors (1 sub, 2 del) / 3 chars = 1.0
    assert pytest.approx(cer.compute()) == 1.0


def test_cer_batching():
    """Test that CER correctly accumulates across multiple batches."""
    cer = CharacterErrorRate()

    # First batch: "helo" vs "hello" = 1 error, 5 ref chars
    cer.update((["helo"], ["hello"]))
    # Second batch: "word" vs "world" = 1 error, 5 ref chars
    cer.update((["word"], ["world"]))

    # Total: 2 errors / 10 chars = 0.2
    assert pytest.approx(cer.compute()) == 0.2
