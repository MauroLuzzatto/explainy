"""
Global Surrogate Model
----------------------
A global surrogate model is an interpretable model that is trained to approximate the 
predictions of a black box model. We can draw conclusions about the black box model 
by interpreting the surrogate model [1].

Characteristics
===============
- global
- contrastive

Source
======
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. 
https://christophm.github.io/interpretable-ml-book/
"""
import os
import subprocess
import warnings
from typing import Dict

import graphviz
import numpy as np
from sklearn.base import is_classifier, is_regressor  # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from explainy.core.explanation_base import ExplanationBase
from explainy.utils.surrogate_plot import SurrogatePlot
from explainy.utils.surrogate_text import SurrogateText


class SurrogateModelExplanation(ExplanationBase):
    """
    Contrastive, global Explanation (global surrogate model)
    """

    def __init__(
        self,
        X,
        y,
        model,
        number_of_features: int = 4,
        config: Dict = None,
        kind="tree",
    ):
        super(SurrogateModelExplanation, self).__init__(config)
        """
        Init the specific explanation class, the base class is "Explanation"

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            sparse (bool): boolean value to generate sparse or non sparse explanation
            show_rating (bool):
            save (bool, optional): boolean value to save the plots. Defaults to True.
           
        Returns:
            None.
        """
        self.X = X
        self.y = y
        self.feature_names = list(X)
        self.model = model
        self.number_of_features = np.log2(number_of_features)
        self.number_of_groups = number_of_features
        self.kind = kind

        kinds = ["tree", "linear"]
        assert (
            self.kind in kinds
        ), f"'{self.kind}' is not a valid option, select from {kinds}"

        natural_language_text_empty = (
            "The features which were most important for the predictions were as"
            " follows: {}"
        )
        method_text_empty = (
            "The feature importance was calculated using a {} surrogate model."
            " {} tree nodes are shown."
        )
        sentence_text_empty = "The samples got a value of {:.2f} if {}"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )

        self.explanation_name = "surrogate"
        self.logger = self.setup_logger(self.explanation_name)

        self._setup()

    def _calculate_importance(self, max_leaf_nodes=100):
        """
        Train a surrogate model (Decision Tree) on the predicted values
        from the original model
        """

        if self.kind == "tree":

            if is_regressor(self.model):
                estimator = DecisionTreeRegressor
            elif is_classifier(self.model):
                estimator = DecisionTreeClassifier

            self.tree_surrgate(estimator, max_leaf_nodes=max_leaf_nodes)

        elif self.kind == "linear":
            if is_regressor(self.model):
                estimator = LinearRegression
            elif is_classifier(self.model):
                estimator = LogisticRegression

            self.linear_surrogate(estimator)

        else:
            raise

        y_hat = self.model.predict(self.X.values)

        self.surrogate_model.fit(self.X, y_hat)
        self.logger.info(
            "Surrogate Model R2 score: {:.2f}".format(
                self.surrogate_model.score(self.X, y_hat)
            )
        )

    def tree_surrgate(self, estimator, max_leaf_nodes, **kwargs):
        """


        Args:
            estimator (TYPE): DESCRIPTION.
            max_leaf_nodes (TYPE): DESCRIPTION.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """

        self.surrogate_model = estimator(
            max_depth=self.number_of_features, max_leaf_nodes=max_leaf_nodes
        )

    def linear_surrogate(self, estimator, **kwargs):
        """


        Args:
            estimator (TYPE): DESCRIPTION.
            **kwargs (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self.surrogate_model = estimator(**kwargs)

    def get_feature_values():
        pass

    def plot(self, index_sample=None, **kwargs):

        if self.kind == "tree":
            self.plot_tree(index_sample, **kwargs)
        elif self.kind == "linear":
            self.plot_bar(index_sample, **kwargs)
        else:
            raise Exception(f'Value of "kind" is not supported: {self.kind}!')

    def plot_bar(self, sample_index):
        raise NotImplementedError("to be done")

    def plot_tree(self, sample_index=None, precision=2, **kwargs):
        """
        use garphix to plot the decision tree
        """
        surrogatePlot = SurrogatePlot(precision=precision, **kwargs)

        self.dot_file = surrogatePlot(
            model=self.surrogate_model,
            feature_names=self.feature_names,
        )

        name, extension = os.path.splitext(self.plot_name)

        try:
            graphviz.Source(
                self.dot_file,
                filename=os.path.join(self.path_plot, name),
                format=extension.replace(".", ""),
            ).view()

        except subprocess.CalledProcessError:
            warnings.warn("plot already open!")

    def save(self, sample_name):

        with open(
            os.path.join(self.path_plot, "{}.dot".format(self.plot_name)),
            "w",
        ) as file:
            file.write(self.dot_file)

        self.save_csv(sample_name)

    def get_method_text(self):
        """
        Define the method introduction text of the explanation type.

        Returns:
            None.
        """
        return self.method_text_empty.format(
            self.surrogate_model.__class__.__name__,
            self.num_to_str[self.number_of_groups].capitalize(),
        )

    def get_natural_language_text(self):
        """
        Define the natural language output using the feature names and its
        values for this explanation type

        Returns:
            None.
        """
        surrogateText = SurrogateText(
            text=self.sentence_text_empty,
            model=self.surrogate_model,
            X=self.X,
            feature_names=self.feature_names,
        )

        sentences = surrogateText.get_text()
        return self.natural_language_text_empty.format(sentences)

    def _setup(self):
        """
        Calculate the feature importance and create the text once

        Returns:
            None.
        """

        self._calculate_importance()
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()
        self.plot_name = self.get_plot_name()

    def explain(self, sample_index, sample_name=None, separator='\n'):
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample (int): number of the sample to create the explanation for

        Returns:
            None.
        """
        if not sample_name:
            sample_name = sample_index

        self.get_prediction(sample_index)
        self.score_text = self.get_score_text()
        self.explanation = self.get_explanation(separator)
        return self.explanation
