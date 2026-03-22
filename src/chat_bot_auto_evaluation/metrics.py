from bert_score import score


class Metrics:
    """
    The `Metrics` class contains methods for calculating perplexity and BERT score for a given set of candidate and reference texts.
    """

    def __init__(self, candidates, references):
        """
        Args:
            candidates (list): candidates list
            references (list): reference list
        """
        self.candidates = candidates
        self.references = references

    def perplexity(self):
        # to add
        pass

    def bert_score(self):
        """
        The `bert_score` function calculates the BERT score between candidate and reference text in
        Polish language with verbose output.
        :return: The `bert_score` method is returning the BERTScore between the candidates and
        references using the Polish language model with verbose output enabled.
        """
        return score(self.candidates, self.references, lang="pl", verbose=True)
