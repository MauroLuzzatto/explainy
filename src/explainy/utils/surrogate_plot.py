import re

import sklearn


class GraphvizNotFoundError(Exception):
    pass


class SurrogatePlot(object):
    """
    This class create the graphviz based surrogate plot using the trained sklearn DecisionTree
    """

    def __init__(self, precision=2, impurity=False, rounded=True, class_names=True):
        self.precision = precision
        self.impurity = impurity
        self.rounded = rounded
        self.class_names = class_names

    def get_plot(self, model, feature_names):
        """
        Update the dot file as desired, simplify the text in the boxes

        Args:
            model (TYPE): DESCRIPTION.
            feature_names (TYPE): DESCRIPTION.

        Returns:
            f (TYPE): DESCRIPTION.

        """
        f = sklearn.tree.export_graphviz(
            model,
            feature_names=feature_names,
            impurity=self.impurity,
            rounded=self.rounded,
            precision=self.precision,
            class_names=self.class_names,
        )
        f = self.one_hot_encoding_text(f)
        return f

    @staticmethod
    def one_hot_encoding_text(f):
        """
        customize the labels text for one-hot encoded features

        Args:
            f (TYPE): DESCRIPTION.

        Returns:
            f (TYPE): DESCRIPTION.
        """
        values = re.findall(r'\[label="(.*?)"\]', f, re.DOTALL)
        for value in values:
            if " - " in value:
                text = value.split("<=")[0].strip()
                feature_name = text.split(" - ")[0]
                feature_value = text.split(" - ")[1]

                node = value.split("\\n")
                new_text = f"{feature_name} is not {feature_value}"
                node[0] = new_text
                f = f.replace(value, "\n".join(node))

        return f

    def __call__(self, model, feature_names):
        return self.get_plot(model, feature_names)
