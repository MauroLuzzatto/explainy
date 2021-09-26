"""
SHAP Explanation
----------------
A prediction can be explained by assuming that each feature value of  the instance is a "player" in a game where 
the prediction is the payout.  Shapley values (a method from coalitional game theory) tells us how  to fairly 
distribute the "payout" among the features. The Shapley value is the average marginal contribution of a feature 
value across all possible coalitions [1].

Characteristics
===============
- local
- non-contrastive

Source
======
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. 
https://christophm.github.io/interpretable-ml-book/
"""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn

from explainy.explanation.explanation_base import ExplanationBase


class ShapExplanation(ExplanationBase):
    """
    Non-contrastive, local Explanation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: sklearn.base.BaseEstimator,
        number_of_features: int = 4,
        config: Dict = None,
    ) -> None:
        super(ShapExplanation, self).__init__(config)
        """
        Init the specific explanation class, the base class is "Explanation"

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            number_of_features (int):
            config

        Returns:
            None.
        """
        self.X = X
        self.y = y
        self.model = model
        self.feature_names = list(self.X)
        self.number_of_features = self.get_number_of_features(
            number_of_features
        )

        natural_language_text_empty = (
            "The {} features which were most important for this particular"
            " sample were (from highest to lowest): {}."
        )
        method_text_empty = (
            "The feature importance was calculated using the SHAP method."
        )
        sentence_text_empty = "'{}' ({:.2f})"

        self.natural_language_text_empty = self.config.get(
            "natural_language_text_empty", natural_language_text_empty
        )
        self.method_text_empty = self.config.get(
            "method_text_empty", method_text_empty
        )
        self.sentence_text_empty = self.config.get(
            "sentence_text_empty", sentence_text_empty
        )

        self.explanation_name = "shap"
        self.logger = self.setup_logger(self.explanation_name)

    def _calculate_importance(self) -> None:
        """
        Explain model predictions using SHAP library

        Returns:
            None.

        """
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X)

        if isinstance(self.explainer.expected_value, np.ndarray):
            self.explainer.expected_value = self.explainer.expected_value[0]

        assert isinstance(
            self.explainer.expected_value, float
        ), "self.explainer.expected_value has wrong type"

    def get_feature_values(self, sample_index: int = 0):
        """
        extract the feature name and its importance per sample

        Args:
            sample_index (int, optional): sample for which the explanation should
            be returned. Defaults to 0.

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        # get absolute values to get the strongst postive and negative contribution
        indexes = np.argsort(abs(self.shap_values[sample_index, :]))
        feature_values = []
        # sort by importance -> highst to lowest
        for index in indexes.tolist()[::-1][: self.number_of_features]:
            feature_values.append(
                (
                    self.feature_names[index],
                    self.shap_values[sample_index, index],
                )
            )
        return feature_values

    def get_score(self, sample_index: int = 0) -> float:
        """
        calculate the overall score of the sample (output-values)

        Args:
            sample_index (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None.
        """
        return (
            np.sum(self.shap_values[sample_index, :])
            + self.explainer.expected_value
        )

    def plot(self, sample_index: int = 0, kind="bar") -> None:
        """


        Args:
            sample_index (int, optional): DESCRIPTION. Defaults to 0.
            kind (TYPE, optional): DESCRIPTION. Defaults to "bar".

        Returns:
            None: DESCRIPTION.

        """
        if kind == "bar":
            self.fig = self.bar_plot(sample_index)
        elif kind == "shap":
            self.fig = self.shap_plot(sample_index)
        else:
            raise

    def bar_plot(self, sample_index: int = 0):
        """
        Create a bar plot of the shape values for a selected sample

        Args:
            sample_index (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None

        """
        indexes = np.argsort(abs(self.shap_values[sample_index, :]))
        sorted_idx = indexes.tolist()[::-1][: self.number_of_features]

        width = self.shap_values[sample_index, sorted_idx]
        y = np.arange(self.number_of_features, 0, -1)
        labels = [self.feature_names[i] for i in sorted_idx]

        fig = plt.figure(
            figsize=(6, max(2, int(0.5 * self.number_of_features)))
        )
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel("Shap Values")
        plt.tight_layout()
        plt.show()
        return fig

    def shap_plot(self, sample_index: int = 0) -> None:
        """
        visualize the first prediction's explanation

        Args:
            sample_index (int, optional): sample for which the explanation should
                be returned. Defaults to 0.
        Returns:
            None.
        """
        shap.force_plot(
            base_value=self.explainer.expected_value,
            shap_values=np.around(
                self.shap_values[sample_index, :], decimals=2
            ),
            features=self.X.iloc[sample_index, :],
            matplotlib=True,
            show=False,
        )

        fig = plt.gcf()
        fig.set_figheight(4)
        fig.set_figwidth(8)
        plt.show()
        return fig

    def log_output(self, sample_index: int) -> None:
        """
        Log the prediction values of the sample

        Args:
            sample (int): DESCRIPTION.

        Returns:
            None.
        """
        self.logger.debug(
            "The expected_value was: {:.2f}".format(
                self.explainer.expected_value
            )
        )
        self.logger.debug(
            "The y_value was: {}".format(self.y.values[sample_index][0])
        )
        self.logger.debug("The predicted value was: {}".format(self.prediction))

    def _setup(self, sample_index, sample_name):
        """


        Args:
            sample_index (TYPE): DESCRIPTION.
            sample_name (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self._calculate_importance()
        self.log_output(sample_index)
        self.feature_values = self.get_feature_values(sample_index)

        self.sentences = self.get_sentences(
            self.feature_values, self.sentence_text_empty
        )
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()
        self.plot_name = self.get_plot_name(sample_name)

    def explain(self, sample_index, sample_name=None, separator="\n") -> None:
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample_index (int): number of the sample to create the explanation for

        Returns:
            None.
        """

        if not sample_name:
            sample_name = sample_index

        self.get_prediction(sample_index)
        self._setup(sample_index, sample_name)

        self.score_text = self.get_score_text()
        self.explanation = self.get_explanation(separator)
        return self.explanation
