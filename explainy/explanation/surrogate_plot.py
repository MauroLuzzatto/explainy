# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:15:31 2020

@author: mauro
"""

import re
import sklearn


class SurrogatePlot(object):
    """
    This class create the graphviz based surrogate plot using the trained sklearn DecisionTree
    """
    
    def __init__(self, precision=2, impurity=False, rounded=True, class_names=True):
        self.precision = precision
        self.impurity = impurity
        self.rounded = rounded
        self.class_names = class_names

    def get_plot(
        self, model, feature_names
    ):
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

                node = value.split('\\n')
                new_text = f"{feature_name} is not {feature_value}"
                node[0] = new_text
                f = f.replace(value, '\n'.join(node))

        return f

    # @staticmethod
    # def simplify_plot(f):

    #     # remove "value = xy" for all cells, except the lowest ones
    #     values = re.findall(r"value = (\d{0,5}\.\d{0,5})", f)
    #     print(values)
    #     for idx, value in enumerate(values):
    #         print(idx, value, len(values))
    #         if (
    #             (len(values) > 3 and idx in [0, 1, 4])
    #             or (len(values) == 7 and idx in [0])
    #             or (len(values) == 15 and idx in [0, 1, 2, 4, 5, 9, 12])
    #         ):
    #             if len(values) == 15:
    #                 pass
    #             else:
    #                 f = re.sub(r"value = {}".format(value), "", f)

    #     f = re.sub(r"value =", "Average rating:\n", f)
    #     print(f)
    #     return f

    # @staticmethod
    # def remove_text(f):
    #     # change the string via regex
    #     f = re.sub(r"(\\nsamples = \d{0,5})", "", f)
    #     f = re.sub(r"(samples = \d{0,5})", "", f)
    #     # f = re.sub(r"(\\n\\n)", "\\n", f)
    #     f = re.sub(r"(\\nvalue)", "value", f)
    #     # f = re.sub(r"<=", "<", f)
    #     return f

    # @staticmethod
    # def add_labels(f):
    #     """
    #     Add True and False labels to the edges

    #     Args:
    #         f (TYPE): DESCRIPTION.

    #     Returns:
    #         f (TYPE): DESCRIPTION.

    #     """

    #     def get_label_based_on_nodes(idx):

    #         true_list = []
    #         label = idx in true_list
    #         return label

    #     matches = re.findall(r"\d -> \d ;", f)
    #     for idx, match in enumerate(matches):
    #         # check if even or not, give label based on this
    #         label = bool(idx % 2 == 0)

    #         first_number = re.match(r"(\d) -> \d ;", match).groups()[0]
    #         second_number = re.match(r"\d -> (\d) ;", match).groups()[0]

    #         if not label:
    #             label = f"{label}"

    #         f = re.sub(
    #             r"{} -> {} ;".format(first_number, second_number),
    #             r'{} -> {} [headlabel="{}     "] ;'.format(
    #                 first_number, second_number, label
    #             ),
    #             f,
    #         )
    #     return f

    def __call__(self, model, feature_names):
        return self.get_plot(model, feature_names)
