import numpy as np
from sklearn.base import is_classifier

from explainy.utils.utils import join_text_with_comma_and_and


class SurrogateText:
    """"""

    def __init__(self, text: str, model: object, X: np.array, feature_names: list):
        """Class to generate text explanation from Decision Trees

        Args:
            text (TYPE): DESCRIPTION.
            model (TYPE): DESCRIPTION.
            X (TYPE): DESCRIPTION.
            feature_names (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self.text = text
        self.model = model
        self.X = X
        self.feature_names = feature_names

        self.children_left = self.model.tree_.children_left
        self.children_right = self.model.tree_.children_right
        self.feature = self.model.tree_.feature
        self.threshold = self.model.tree_.threshold

        if is_classifier(self.model):
            self.values = np.argmax(self.model.tree_.value, axis=2).reshape(
                self.model.tree_.value.shape[0], 1
            )
        else:
            self.values = self.model.tree_.value.reshape(
                self.model.tree_.value.shape[0], 1
            )

    def get_text(self):
        """Returns:
        TYPE: DESCRIPTION.

        """
        paths = self.get_paths()

        texts = []
        for key in paths:
            sentences = self.get_rule(paths[key])
            sentences = join_text_with_comma_and_and(sentences)
            score = self.values[key][0]
            texts.append(self.text.format(score, sentences))

        return " ".join([text + "." for text in texts])

    def get_paths(self):
        """Returns:
        None.

        """
        # Leaves
        leave_id = self.model.apply(self.X)

        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

        return paths

    def find_path(self, node_numb, path, x):
        """Args:
            node_numb (TYPE): DESCRIPTION.
            path (TYPE): DESCRIPTION.
            x (TYPE): DESCRIPTION.

        Returns:
            bool: DESCRIPTION.

        """
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False

        if self.children_left[node_numb] != -1:
            left = self.find_path(self.children_left[node_numb], path, x)

        if self.children_right[node_numb] != -1:
            right = self.find_path(self.children_right[node_numb], path, x)

        if left or right:
            return True

        path.remove(node_numb)
        return False

    def get_rule(self, path):
        """Args:
            path (TYPE): DESCRIPTION.

        Returns:
            TYPE: DESCRIPTION.

        """
        mask = []
        for index, node in enumerate(path):
            # check if we are not in the leaf
            if index != len(path) - 1:
                feature_name_per_node = self.feature_names[self.feature[node]]

                one_hot_feature_bool = " - " in feature_name_per_node
                if one_hot_feature_bool:
                    feature_name, feature_value = feature_name_per_node.split(" - ")

                # if under the threshold
                if self.children_left[node] == path[index + 1]:
                    if one_hot_feature_bool:
                        text = f"{feature_name}' was not '{feature_value}'"
                    else:
                        text = (
                            f"'{feature_name_per_node}' was less or equal than"
                            f" {self.threshold[node]:.2f}"
                        )
                else:
                    if one_hot_feature_bool:
                        text = f"'{feature_name}' was '{feature_value}'"
                    else:
                        text = (
                            f"'{feature_name_per_node}' was greater than"
                            f" {self.threshold[node]:.2f}"
                        )
                mask.append(text)

        sentences = [text for text in mask if text]
        return sentences
