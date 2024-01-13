from typing import Tuple, Union


class Explanation:
    """Explanation class"""

    def __init__(
        self,
        score_text: str,
        method_text: str,
        natural_language_text: str,
        separator: str = "\n",
    ):
        """Initiate the Explanation class"""
        self.score_text = score_text
        self.method_text = method_text
        self.natural_language_text = natural_language_text
        self.explanation = self.get_explanation(separator=separator)

    def get_explanation(
        self, separator: str = "\n"
    ) -> Union[str, Tuple[str, str, str]]:
        """Get the final explanation as a string or tuple

        Args:
            separator (str, optional): Seperator to join the sub-explanations. Defaults to "\n".

        Returns:
            Union[str, Tuple[str, str, str]]: final explanation
        """
        explanation = (
            self.score_text,
            self.method_text,
            self.natural_language_text,
        )

        if separator:
            return separator.join(explanation)
        else:
            return explanation

    def print_output(self, separator: str = "\n") -> None:
        print(self.get_explanation(separator))

    def __str__(self) -> str:
        return self.get_explanation(separator="\n")
