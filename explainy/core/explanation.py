class Explanation(object):
    """
    Explanation class
    """

    def __init__(self, score_text, method_text, natural_language_text, separator="\n"):
        """ """
        self.score_text = score_text
        self.method_text = method_text
        self.natural_language_text = natural_language_text
        self.explanation = self.get_explanation(separator=separator)

    def get_explanation(self, separator="\n"):
        if separator:
            explanation = separator.join(
                [self.score_text, self.method_text, self.natural_language_text]
            )
        else:
            explanation = (
                self.score_text,
                self.method_text,
                self.natural_language_text,
            )
        return explanation

    def print_output(self, separator="\n"):
        print(self.get_explanation(separator))

    def __str__(self, separator="\n"):
        return self.get_explanation(separator)
