__all__ = ["CorpusForTest"]


class CorpusForTest:
    def __init__(self, lower_split=False):
        def preproc(text):
            if lower_split:
                return text.lower().split()
            else:
                return text

        # BLEU Paper examples
        self.cand_1 = preproc("the the the the the the the")
        self.ref_1a = preproc("The cat is on the mat")
        self.ref_1b = preproc("There is a cat on the mat")

        self.cand_2a = preproc(
            "It is a guide to action which ensures that the military always obeys the commands of the party"
        )
        self.cand_2b = preproc("It is to insure the troops forever hearing the activity guidebook that " "party direct")
        self.ref_2a = preproc(
            "It is a guide to action that ensures that the military will forever heed " "Party commands"
        )
        self.ref_2b = preproc(
            "It is the guiding principle which guarantees the military forces always being under the command of "
            "the Party"
        )
        self.ref_2c = preproc("It is the practical guide for the army always to heed the directions of the party")

        self.cand_3 = preproc("of the")

        self.references_1 = [self.ref_1a, self.ref_1b]
        self.references_2 = [self.ref_2a, self.ref_2b, self.ref_2c]

        self.sample_1 = ([self.cand_1], [self.references_1])
        self.sample_2 = ([self.cand_3], [self.references_2])
        self.sample_3 = ([self.cand_2a], [self.references_2])
        self.sample_4 = ([self.cand_2b], [self.references_2])
        self.sample_5 = ([self.cand_2a, self.cand_2b], [self.references_2, self.references_2])

        self.references_3 = [self.ref_2a, self.ref_2b]
        self.references_4 = [self.ref_2b, self.ref_2c]
        self.references_5 = [self.ref_2a, self.ref_2c]

        self.chunks = [
            ([self.cand_1], [self.references_1]),
            ([self.cand_2a], [self.references_2]),
            ([self.cand_2b], [self.references_2]),
            ([self.cand_1], [[self.ref_1a]]),
            ([self.cand_2a], [self.references_3]),
            ([self.cand_2b], [self.references_3]),
            ([self.cand_1], [[self.ref_1b]]),
            ([self.cand_2a], [self.references_4]),
            ([self.cand_2b], [self.references_4]),
            ([self.cand_1], [self.references_1]),
            ([self.cand_2a], [self.references_5]),
            ([self.cand_2b], [self.references_5]),
            ([self.cand_1], [[self.ref_1a]]),
            ([self.cand_2a], [[self.ref_2a]]),
            ([self.cand_2b], [[self.ref_2c]]),
        ]
