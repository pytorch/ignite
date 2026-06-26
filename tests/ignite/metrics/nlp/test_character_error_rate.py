import pytest
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import CharacterErrorRate


def test_zero_cer_identical():
    cer = CharacterErrorRate()
    cer.update((["hello world"], ["hello world"]))
    assert cer.compute() == pytest.approx(0.0)


def test_cer_single_deletion():
    cer = CharacterErrorRate()
    cer.update((["helo"], ["hello"]))
    assert cer.compute() == pytest.approx(1 / 5)


def test_cer_single_insertion():
    cer = CharacterErrorRate()
    cer.update((["hello"], ["helo"]))
    assert cer.compute() == pytest.approx(1 / 4)


def test_cer_single_substitution():
    cer = CharacterErrorRate()
    cer.update((["bat"], ["cat"]))
    assert cer.compute() == pytest.approx(1 / 3)


def test_cer_completely_wrong():
    cer = CharacterErrorRate()
    cer.update((["xyz"], ["abc"]))
    assert cer.compute() == pytest.approx(1.0)


def test_cer_empty_prediction():
    cer = CharacterErrorRate()
    cer.update(([""], ["hello"]))
    assert cer.compute() == pytest.approx(1.0)


def test_cer_empty_reference_raises():
    cer = CharacterErrorRate()
    cer.update((["hello world", "hello"], ["hello world", ""]))
    assert cer.compute() == pytest.approx(0.0)


def test_cer_batch():
    cer = CharacterErrorRate()
    cer.update((["hello", "cat"], ["hello", "bat"]))
    assert cer.compute() == pytest.approx(1 / 8)


def test_cer_accumulates_across_updates():
    cer = CharacterErrorRate()
    cer.update((["hello"], ["hello"]))
    cer.update((["cat"], ["bat"]))
    assert cer.compute() == pytest.approx(1 / 8)


def test_cer_reset_clears_state():
    cer = CharacterErrorRate()
    cer.update((["cat"], ["bat"]))
    cer.reset()
    cer.update((["hello"], ["hello"]))
    assert cer.compute() == pytest.approx(0.0)


def test_cer_single_string_input():
    cer = CharacterErrorRate()
    cer.update(("helo", "hello"))
    assert cer.compute() == pytest.approx(1 / 5)


def test_cer_whitespace_counts_as_character():
    cer = CharacterErrorRate()
    cer.update((["ab"], ["a b"]))
    assert cer.compute() == pytest.approx(1 / 3)


def test_cer_not_computable_before_update():
    cer = CharacterErrorRate()
    with pytest.raises(NotComputableError):
        cer.compute()


def test_cer_multiline():
    cer = CharacterErrorRate()
    cer.update((["hello\nworld"], ["hello\nworld"]))
    assert cer.compute() == pytest.approx(0.0)


def test_cer_unicode():
    cer = CharacterErrorRate()
    cer.update((["cafe"], ["café"]))
    assert cer.compute() == pytest.approx(1 / 4)
