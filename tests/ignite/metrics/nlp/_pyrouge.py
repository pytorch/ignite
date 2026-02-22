# Vendored from py-rouge 1.1 (https://github.com/Diego999/py-rouge)
# Original license: Apache 2.0
# Modifications: removed pkg_resources dependency; data files resolved via __file__;
#                wordnet/stemmer loading made conditional on stemming=True.
import collections
import itertools
import os
import re

import nltk


class Rouge:
    DEFAULT_METRICS = {"rouge-n"}
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n", "rouge-l", "rouge-w"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {"words", "bytes"}
    REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9]")

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the official ROUGE script
    KEEP_CANNOT_IN_ONE_WORD = re.compile("cannot")
    KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile("_cannot_")

    WORDNET_KEY_VALUE = {}
    WORDNET_DB_FILEPATH = "wordnet_key_value.txt"
    WORDNET_DB_FILEPATH_SPECIAL_CASE = "wordnet_key_value_special_cases.txt"
    WORDNET_DB_DELIMITER = "|"
    STEMMER = None

    def __init__(
        self,
        metrics=None,
        max_n=None,
        limit_length=True,
        length_limit=665,
        length_limit_type="bytes",
        apply_avg=True,
        apply_best=False,
        stemming=True,
        alpha=0.5,
        weight_factor=1.0,
        ensure_compatibility=True,
    ):
        self.metrics = metrics[:] if metrics is not None else Rouge.DEFAULT_METRICS
        for m in self.metrics:
            if m not in Rouge.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        if self.max_n is not None:
            index_rouge_n = self.metrics.index("rouge-n")
            del self.metrics[index_rouge_n]
            self.metrics += ["rouge-{}".format(n) for n in range(1, self.max_n + 1)]
        self.metrics = set(self.metrics)

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in Rouge.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError("Unknown length_limit_type '{}'".format(length_limit_type))

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.limit_length = False
        self.length_limit_type = length_limit_type
        self.stemming = stemming

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.weight_factor = weight_factor
        if self.weight_factor <= 0:
            raise ValueError("ROUGE-W weight factor must greater than 0.")
        self.ensure_compatibility = ensure_compatibility

        # Only load stemming resources when stemming is actually enabled
        if self.stemming:
            if len(Rouge.WORDNET_KEY_VALUE) == 0:
                Rouge.load_wordnet_db(ensure_compatibility)
            if Rouge.STEMMER is None:
                Rouge.load_stemmer(ensure_compatibility)

    @staticmethod
    def load_stemmer(ensure_compatibility):
        Rouge.STEMMER = (
            nltk.stem.porter.PorterStemmer("ORIGINAL_ALGORITHM")
            if ensure_compatibility
            else nltk.stem.porter.PorterStemmer()
        )

    @staticmethod
    def load_wordnet_db(ensure_compatibility):
        files_to_load = [Rouge.WORDNET_DB_FILEPATH]
        if ensure_compatibility:
            files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

        _data_dir = os.path.dirname(os.path.abspath(__file__))
        for wordnet_db in files_to_load:
            filepath = os.path.join(_data_dir, wordnet_db)
            if not os.path.exists(filepath):
                raise FileNotFoundError("The file '{}' does not exist".format(filepath))

            with open(filepath, "r", encoding="utf-8") as fp:
                for line in fp:
                    k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                    assert k not in Rouge.WORDNET_KEY_VALUE
                    Rouge.WORDNET_KEY_VALUE[k] = v

    @staticmethod
    def tokenize_text(text, language="english"):
        return nltk.word_tokenize(text, language)

    @staticmethod
    def split_into_sentences(text, ensure_compatibility, language="english"):
        if ensure_compatibility:
            return text.split("\n")
        else:
            return nltk.sent_tokenize(text, language)

    @staticmethod
    def stem_tokens(tokens):
        for i, token in enumerate(tokens):
            if len(token) > 0:
                if len(token) > 3:
                    if token in Rouge.WORDNET_KEY_VALUE:
                        token = Rouge.WORDNET_KEY_VALUE[token]
                    else:
                        token = Rouge.STEMMER.stem(token)
                    tokens[i] = token
        return tokens

    @staticmethod
    def _get_ngrams(n, text):
        ngram_set = collections.defaultdict(int)
        max_index_ngram_start = len(text) - n
        for i in range(max_index_ngram_start + 1):
            ngram_set[tuple(text[i : i + n])] += 1
        return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        return list(itertools.chain(*[_.split() for _ in sentences]))

    @staticmethod
    def _get_word_ngrams_and_length(n, sentences):
        assert len(sentences) > 0
        assert n > 0
        tokens = Rouge._split_into_words(sentences)
        return Rouge._get_ngrams(n, tokens), tokens, len(tokens) - (n - 1)

    @staticmethod
    def _get_unigrams(sentences):
        assert len(sentences) > 0
        tokens = Rouge._split_into_words(sentences)
        unigram_set = collections.defaultdict(int)
        for token in tokens:
            unigram_set[token] += 1
        return unigram_set, len(tokens)

    @staticmethod
    def _compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=0.5, weight_factor=1.0):
        precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
        if weight_factor != 1.0:
            precision = precision ** (1.0 / weight_factor)
        recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
        if weight_factor != 1.0:
            recall = recall ** (1.0 / weight_factor)
        f1_score = Rouge._compute_f_score(precision, recall, alpha)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_f_score(precision, recall, alpha=0.5):
        return (
            0.0
            if (recall == 0.0 or precision == 0.0)
            else precision * recall / ((1 - alpha) * precision + alpha * recall)
        )

    @staticmethod
    def _compute_ngrams(evaluated_sentences, reference_sentences, n):
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, _, evaluated_count = Rouge._get_word_ngrams_and_length(n, evaluated_sentences)
        reference_ngrams, _, reference_count = Rouge._get_word_ngrams_and_length(n, reference_sentences)

        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    @staticmethod
    def _compute_ngrams_lcs(evaluated_sentences, reference_sentences, weight_factor=1.0):
        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = "|"
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"

            return vals, dirs

        def _wlcs(x, y, weight_factor):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(float)
            dirs = collections.defaultdict(int)
            lengths = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        length_tmp = lengths[i - 1, j - 1]
                        vals[i, j] = vals[i - 1, j - 1] + (length_tmp + 1) ** weight_factor - length_tmp**weight_factor
                        dirs[i, j] = "|"
                        lengths[i, j] = length_tmp + 1
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = "^"
                        lengths[i, j] = 0
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = "<"
                        lengths[i, j] = 0

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == "|":
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == "^":
                    m -= 1
                elif dirs[m, n] == "<":
                    n -= 1
                else:
                    raise UnboundLocalError("Illegal move")
            return mask

        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_unigrams_dict, evaluated_count = Rouge._get_unigrams(evaluated_sentences)
        reference_unigrams_dict, reference_count = Rouge._get_unigrams(reference_sentences)

        use_WLCS = weight_factor != 1.0
        if use_WLCS:
            evaluated_count = evaluated_count**weight_factor
            reference_count = 0

        overlapping_count = 0.0
        for reference_sentence in reference_sentences:
            reference_sentence_tokens = reference_sentence.split()
            if use_WLCS:
                reference_count += len(reference_sentence_tokens) ** weight_factor
            hit_mask = [0 for _ in range(len(reference_sentence_tokens))]

            for evaluated_sentence in evaluated_sentences:
                evaluated_sentence_tokens = evaluated_sentence.split()

                if use_WLCS:
                    _, lcs_dirs = _wlcs(reference_sentence_tokens, evaluated_sentence_tokens, weight_factor)
                else:
                    _, lcs_dirs = _lcs(reference_sentence_tokens, evaluated_sentence_tokens)
                _mark_lcs(hit_mask, lcs_dirs, len(reference_sentence_tokens), len(evaluated_sentence_tokens))

            overlapping_count_length = 0
            for ref_token_id, val in enumerate(hit_mask):
                if val == 1:
                    token = reference_sentence_tokens[ref_token_id]
                    if evaluated_unigrams_dict[token] > 0 and reference_unigrams_dict[token] > 0:
                        evaluated_unigrams_dict[token] -= 1
                        reference_unigrams_dict[ref_token_id] -= 1

                        if use_WLCS:
                            overlapping_count_length += 1
                            if (
                                ref_token_id + 1 < len(hit_mask) and hit_mask[ref_token_id + 1] == 0
                            ) or ref_token_id + 1 == len(hit_mask):
                                overlapping_count += overlapping_count_length**weight_factor
                                overlapping_count_length = 0
                        else:
                            overlapping_count += 1

        if use_WLCS:
            reference_count = reference_count**weight_factor

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self, hypothesis, references):
        if isinstance(hypothesis, str):
            hypothesis, references = [hypothesis], [references]

        if type(hypothesis) != type(references):
            raise ValueError("'hyps' and 'refs' are not of the same type")

        if len(hypothesis) != len(references):
            raise ValueError("'hyps' and 'refs' do not have the same length")

        scores = {}
        has_rouge_n_metric = len([metric for metric in self.metrics if metric.split("-")[-1].isdigit()]) > 0
        if has_rouge_n_metric:
            scores = {**scores, **self._get_scores_rouge_n(hypothesis, references)}

        has_rouge_l_metric = len([metric for metric in self.metrics if metric.split("-")[-1].lower() == "l"]) > 0
        if has_rouge_l_metric:
            scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, False)}

        has_rouge_w_metric = len([metric for metric in self.metrics if metric.split("-")[-1].lower() == "w"]) > 0
        if has_rouge_w_metric:
            scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, True)}

        return scores

    def _get_scores_rouge_n(self, all_hypothesis, all_references):
        metrics = [metric for metric in self.metrics if metric.split("-")[-1].isdigit()]

        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in Rouge.STATS} for metric in metrics}
        else:
            scores = {
                metric: [{stat: [] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))] for metric in metrics
            }

        for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
            references = (
                [self._preprocess_summary_as_a_whole(reference) for reference in references]
                if has_multiple_references
                else [self._preprocess_summary_as_a_whole(references)]
            )

            for metric in metrics:
                suffix = metric.split("-")[-1]
                n = int(suffix)

                if self.apply_avg:
                    total_hypothesis_ngrams_count = 0
                    total_reference_ngrams_count = 0
                    total_ngrams_overlapping_count = 0

                    for reference in references:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(
                            hypothesis, reference, n
                        )
                        total_hypothesis_ngrams_count += hypothesis_count
                        total_reference_ngrams_count += reference_count
                        total_ngrams_overlapping_count += overlapping_ngrams

                    score = Rouge._compute_p_r_f_score(
                        total_hypothesis_ngrams_count,
                        total_reference_ngrams_count,
                        total_ngrams_overlapping_count,
                        self.alpha,
                    )

                    for stat in Rouge.STATS:
                        scores[metric][stat] += score[stat]
                else:
                    if self.apply_best:
                        best_current_score = None
                        for reference in references:
                            hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(
                                hypothesis, reference, n
                            )
                            score = Rouge._compute_p_r_f_score(
                                hypothesis_count, reference_count, overlapping_ngrams, self.alpha
                            )
                            if best_current_score is None or score["r"] > best_current_score["r"]:
                                best_current_score = score

                        for stat in Rouge.STATS:
                            scores[metric][stat] += best_current_score[stat]
                    else:
                        for reference in references:
                            hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(
                                hypothesis, reference, n
                            )
                            score = Rouge._compute_p_r_f_score(
                                hypothesis_count, reference_count, overlapping_ngrams, self.alpha
                            )
                            for stat in Rouge.STATS:
                                scores[metric][sample_id][stat].append(score[stat])

        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for metric in metrics:
                for stat in Rouge.STATS:
                    scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _get_scores_rouge_l_or_w(self, all_hypothesis, all_references, use_w=False):
        metric = "rouge-w" if use_w else "rouge-l"
        if self.apply_avg or self.apply_best:
            scores = {metric: {stat: 0.0 for stat in Rouge.STATS}}
        else:
            scores = {metric: [{stat: [] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))]}

        for sample_id, (hypothesis_sentences, references_sentences) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis_sentences, str)
            has_multiple_references = False
            if isinstance(references_sentences, list):
                has_multiple_references = len(references_sentences) > 1
                if not has_multiple_references:
                    references_sentences = references_sentences[0]

            hypothesis_sentences = self._preprocess_summary_per_sentence(hypothesis_sentences)
            references_sentences = (
                [self._preprocess_summary_per_sentence(reference) for reference in references_sentences]
                if has_multiple_references
                else [self._preprocess_summary_per_sentence(references_sentences)]
            )

            if self.apply_avg:
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0

                for reference_sentences in references_sentences:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(
                        hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0
                    )
                    total_hypothesis_ngrams_count += hypothesis_count
                    total_reference_ngrams_count += reference_count
                    total_ngrams_overlapping_count += overlapping_ngrams

                score = Rouge._compute_p_r_f_score(
                    total_hypothesis_ngrams_count,
                    total_reference_ngrams_count,
                    total_ngrams_overlapping_count,
                    self.alpha,
                    self.weight_factor,
                )

                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            else:
                if self.apply_best:
                    best_current_score = None
                    best_current_score_wlcs = None
                    for reference_sentences in references_sentences:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(
                            hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0
                        )
                        score = Rouge._compute_p_r_f_score(
                            hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor
                        )

                        if use_w:
                            reference_count_for_score = reference_count ** (1.0 / self.weight_factor)
                            overlapping_ngrams_for_score = overlapping_ngrams
                            score_wlcs = (overlapping_ngrams_for_score / reference_count_for_score) ** (
                                1.0 / self.weight_factor
                            )

                            if best_current_score_wlcs is None or score_wlcs > best_current_score_wlcs:
                                best_current_score = score
                                best_current_score_wlcs = score_wlcs
                        else:
                            if best_current_score is None or score["r"] > best_current_score["r"]:
                                best_current_score = score

                    for stat in Rouge.STATS:
                        scores[metric][stat] += best_current_score[stat]
                else:
                    for reference_sentences in references_sentences:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(
                            hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0
                        )
                        score = Rouge._compute_p_r_f_score(
                            hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor
                        )

                        for stat in Rouge.STATS:
                            scores[metric][sample_id][stat].append(score[stat])

        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for stat in Rouge.STATS:
                scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _preprocess_summary_as_a_whole(self, summary):
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        if self.limit_length:
            if self.length_limit_type == "words":
                summary = " ".join(sentences)
                all_tokens = summary.split()
                summary = " ".join(all_tokens[: self.length_limit])
            elif self.length_limit_type == "bytes":
                summary = ""
                current_len = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)

                    if current_len + sentence_len < self.length_limit:
                        if current_len != 0:
                            summary += " "
                        summary += sentence
                        current_len += sentence_len
                    else:
                        if current_len > 0:
                            summary += " "
                        summary += sentence[: self.length_limit - current_len]
                        break
        else:
            summary = " ".join(sentences)

        summary = Rouge.REMOVE_CHAR_PATTERN.sub(" ", summary.lower()).strip()

        if self.ensure_compatibility:
            tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub("_cannot_", summary))
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(" ", summary))

        if self.stemming:
            self.stem_tokens(tokens)

        if self.ensure_compatibility:
            preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub("cannot", " ".join(tokens))]
        else:
            preprocessed_summary = [" ".join(tokens)]

        return preprocessed_summary

    def _preprocess_summary_per_sentence(self, summary):
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        if self.limit_length:
            final_sentences = []
            current_len = 0
            if self.length_limit_type == "words":
                for sentence in sentences:
                    tokens = sentence.strip().split()
                    tokens_len = len(tokens)
                    if current_len + tokens_len < self.length_limit:
                        sentence = " ".join(tokens)
                        final_sentences.append(sentence)
                        current_len += tokens_len
                    else:
                        sentence = " ".join(tokens[: self.length_limit - current_len])
                        final_sentences.append(sentence)
                        break
            elif self.length_limit_type == "bytes":
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)
                    if current_len + sentence_len < self.length_limit:
                        final_sentences.append(sentence)
                        current_len += sentence_len
                    else:
                        sentence = sentence[: self.length_limit - current_len]
                        final_sentences.append(sentence)
                        break
            sentences = final_sentences

        final_sentences = []
        for sentence in sentences:
            sentence = Rouge.REMOVE_CHAR_PATTERN.sub(" ", sentence.lower()).strip()

            if self.ensure_compatibility:
                tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub("_cannot_", sentence))
            else:
                tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(" ", sentence))

            if self.stemming:
                self.stem_tokens(tokens)

            if self.ensure_compatibility:
                sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub("cannot", " ".join(tokens))
            else:
                sentence = " ".join(tokens)

            final_sentences.append(sentence)

        return final_sentences
