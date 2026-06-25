import pytest
from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import CharacterErrorRate


class TestCharacterErrorRate:
    def test_zero_cer_identical(self):
        """Identical prediction and reference → CER = 0."""
        cer = CharacterErrorRate()
        cer.update((["hello world"], ["hello world"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_single_deletion(self):
        """One character deleted from prediction."""
        cer = CharacterErrorRate()
        cer.update((["helo"], ["hello"]))
        assert cer.compute() == pytest.approx(1 / 5)

    def test_cer_single_insertion(self):
        """One character inserted into prediction."""
        cer = CharacterErrorRate()
        cer.update((["hello"], ["helo"]))
        assert cer.compute() == pytest.approx(1 / 4)

    def test_cer_single_substitution(self):
        """One character substituted."""
        cer = CharacterErrorRate()
        cer.update((["bat"], ["cat"]))
        assert cer.compute() == pytest.approx(1 / 3)

    def test_cer_completely_wrong(self):
        """Prediction shares no characters with reference."""
        cer = CharacterErrorRate()
        cer.update((["xyz"], ["abc"]))
        assert cer.compute() == pytest.approx(1.0)

    def test_cer_empty_prediction(self):
        """Empty prediction → CER = 1.0 (all chars deleted)."""
        cer = CharacterErrorRate()
        cer.update(([""], ["hello"]))
        assert cer.compute() == pytest.approx(1.0)

    def test_cer_empty_reference_raises(self):
        """Empty reference → raises NotComputableError."""
        cer = CharacterErrorRate()
        cer.update((["hello"], [""]))
        with pytest.raises(NotComputableError):
            cer.compute()

    def test_cer_batch(self):
        """Batch of two sequences accumulated correctly."""
        cer = CharacterErrorRate()
        cer.update((["hello", "cat"], ["hello", "bat"]))
        assert cer.compute() == pytest.approx(1 / 8)

    def test_cer_accumulates_across_updates(self):
        """Multiple update() calls accumulate correctly."""
        cer = CharacterErrorRate()
        cer.update((["hello"], ["hello"]))
        cer.update((["cat"], ["bat"]))
        assert cer.compute() == pytest.approx(1 / 8)

    def test_cer_reset_clears_state(self):
        """reset() clears accumulated state."""
        cer = CharacterErrorRate()
        cer.update((["cat"], ["bat"]))
        cer.reset()
        cer.update((["hello"], ["hello"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_single_string_input(self):
        """Single string (not list) input handled correctly."""
        cer = CharacterErrorRate()
        cer.update(("helo", "hello"))
        assert cer.compute() == pytest.approx(1 / 5)

    def test_cer_whitespace_counts_as_character(self):
        """Spaces are treated as characters, not word separators."""
        cer = CharacterErrorRate()
        cer.update((["ab"], ["a b"]))
        assert cer.compute() == pytest.approx(1 / 3)

    def test_cer_not_computable_before_update(self):
        """compute() before any update raises NotComputableError."""
        cer = CharacterErrorRate()
        with pytest.raises(NotComputableError):
            cer.compute()

    def test_cer_multiline(self):
        """Newlines treated as characters."""
        cer = CharacterErrorRate()
        cer.update((["hello\nworld"], ["hello\nworld"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_unicode(self):
        """Unicode characters handled correctly."""
        cer = CharacterErrorRate()
        cer.update((["cafe"], ["café"]))
        assert cer.compute() == pytest.approx(1 / 4)
