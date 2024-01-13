import re
from typing import List

import sklearn

from explainy.utils.typing import ModelType


class GraphvizNotFoundError(Exception):
    pass


class SurrogatePlot:
    """This class create the graphviz based surrogate plot using the trained sklearn DecisionTree"""

    def __init__(
        self,
        precision: int = 2,
        impurity: bool = False,
        rounded: bool = True,
        class_names: bool = True,
    ):
        self.precision = precision
        self.impurity = impurity
        self.rounded = rounded
        self.class_names = class_names

    def get_plot(self, model: ModelType, feature_names: List[str]):
        """Update the dot file as desired, simplify the text in the boxes

        Args:
            model (TYPE): DESCRIPTION.
            feature_names (TYPE): DESCRIPTION.

        Returns:
            f (TYPE): DESCRIPTION.

        """
        tree = sklearn.tree.export_graphviz(
            model,
            feature_names=feature_names,
            impurity=self.impurity,
            rounded=self.rounded,
            precision=self.precision,
            class_names=self.class_names,
        )
        tree = self.one_hot_encoding_text(tree)
        return tree

    @staticmethod
    def one_hot_encoding_text(tree: str) -> str:
        """Customize the labels text for one-hot encoded features

        Args:
            f (TYPE): DESCRIPTION.

        Returns:
            f (TYPE): DESCRIPTION.
        """
        values = re.findall(r'\[label="(.*?)"\]', tree, re.DOTALL)
        for value in values:
            if " - " in value:
                text = value.split("<=")[0].strip()
                feature_name = text.split(" - ")[0]
                feature_value = text.split(" - ")[1]
                node = value.split("\\n")
                new_text = f"{feature_name} is not {feature_value}"
                node[0] = new_text
                tree = tree.replace(value, "\n".join(node))
        return tree

    def __call__(self, model: ModelType, feature_names: List[str]):
        return self.get_plot(model, feature_names)
