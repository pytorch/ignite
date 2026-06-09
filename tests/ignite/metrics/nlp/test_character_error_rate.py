import pytest



# Mock import path — adjust when placed in actual repo
# from ignite.metrics.nlp import CharacterErrorRate


def _edit_distance(ref, pred):
    """Reference implementation for testing."""
    n, m = len(ref), len(pred)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            dp[j] = prev_diag if ref[i - 1] == pred[j - 1] else min(dp[j - 1], dp[j], prev_diag) + 1
            prev_diag = temp
    return dp[m]


class _MockCER:
    """Standalone mock to validate logic before wiring into ignite."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._num_errors = 0.0
        self._num_refs = 0.0

    def update(self, output):
        y_pred, y = output
        if isinstance(y_pred, str):
            y_pred, y = [y_pred], [y]
        for p, r in zip(y_pred, y):
            p_chars = list(p)
            r_chars = list(r)
            self._num_errors += _edit_distance(r_chars, p_chars)
            self._num_refs += len(r_chars)

    def compute(self):
        if self._num_refs == 0:
            raise ValueError("No references.")
        return self._num_errors / self._num_refs


class TestCharacterErrorRate:

    def test_zero_cer_identical(self):
        """Identical prediction and reference → CER = 0."""
        cer = _MockCER()
        cer.update((["hello world"], ["hello world"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_single_deletion(self):
        """One character deleted from prediction."""
        cer = _MockCER()
        # ref: "hello" (5 chars), pred: "helo" (1 deletion)
        cer.update((["helo"], ["hello"]))
        assert cer.compute() == pytest.approx(1 / 5)

    def test_cer_single_insertion(self):
        """One character inserted into prediction."""
        cer = _MockCER()
        # ref: "helo" (4 chars), pred: "hello" (1 insertion)
        cer.update((["hello"], ["helo"]))
        assert cer.compute() == pytest.approx(1 / 4)

    def test_cer_single_substitution(self):
        """One character substituted."""
        cer = _MockCER()
        # ref: "cat" (3 chars), pred: "bat" (1 substitution)
        cer.update((["bat"], ["cat"]))
        assert cer.compute() == pytest.approx(1 / 3)

    def test_cer_completely_wrong(self):
        """Prediction shares no characters with reference."""
        cer = _MockCER()
        cer.update((["xyz"], ["abc"]))
        assert cer.compute() == pytest.approx(1.0)

    def test_cer_empty_prediction(self):
        """Empty prediction → CER = 1.0 (all chars deleted)."""
        cer = _MockCER()
        cer.update(([""], ["hello"]))
        assert cer.compute() == pytest.approx(1.0)

    def test_cer_empty_reference(self):
        """Empty reference → CER > 1.0 (all chars are insertions)."""
        cer = _MockCER()
        cer.update((["hello"], [""]))
        # refs = 0 for this pair but total refs = 0 → should raise
        with pytest.raises((ValueError, ZeroDivisionError)):
            cer.compute()

    def test_cer_batch(self):
        """Batch of two sequences accumulated correctly."""
        cer = _MockCER()
        # pair 1: "hello" vs "hello" → 0 errors, 5 refs
        # pair 2: "cat" vs "bat"    → 1 error,  3 refs
        cer.update((["hello", "cat"], ["hello", "bat"]))
        assert cer.compute() == pytest.approx(1 / 8)

    def test_cer_accumulates_across_updates(self):
        """Multiple update() calls accumulate correctly."""
        cer = _MockCER()
        cer.update((["hello"], ["hello"]))   # 0 errors, 5 refs
        cer.update((["cat"], ["bat"]))        # 1 error,  3 refs
        assert cer.compute() == pytest.approx(1 / 8)

    def test_cer_reset_clears_state(self):
        """reset() clears accumulated state."""
        cer = _MockCER()
        cer.update((["cat"], ["bat"]))
        cer.reset()
        cer.update((["hello"], ["hello"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_single_string_input(self):
        """Single string (not list) input handled correctly."""
        cer = _MockCER()
        cer.update(("helo", "hello"))
        assert cer.compute() == pytest.approx(1 / 5)

    def test_cer_whitespace_counts_as_character(self):
        """Spaces are treated as characters, not separators."""
        cer = _MockCER()
        # ref: "a b" (3 chars incl space), pred: "ab" (1 deletion)
        cer.update((["ab"], ["a b"]))
        assert cer.compute() == pytest.approx(1 / 3)

    def test_cer_not_computable_before_update(self):
        """compute() before any update raises error."""
        cer = _MockCER()
        with pytest.raises((ValueError, ZeroDivisionError)):
            cer.compute()

    def test_cer_multiline(self):
        """Newlines treated as characters."""
        cer = _MockCER()
        cer.update((["hello\nworld"], ["hello\nworld"]))
        assert cer.compute() == pytest.approx(0.0)

    def test_cer_unicode(self):
        """Unicode characters handled correctly."""
        cer = _MockCER()
        # ref: "café" (4 chars), pred: "cafe" (1 substitution)
        cer.update((["cafe"], ["café"]))
        assert cer.compute() == pytest.approx(1 / 4)
