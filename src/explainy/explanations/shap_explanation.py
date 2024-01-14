"""SHAP Explanation
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

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.typing import Config, ModelType


class ShapExplanation(ExplanationBase):
    """Non-contrastive, local Explanation"""

    explanation_type: str = "local"
    explanation_style: str = "non-contrastive"
    explanation_name: str = "shap"

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: ModelType,
        number_of_features: int = 4,
        config: Optional[Config] = None,
        **kwargs,
    ) -> None:
        super(ShapExplanation, self).__init__(model, config)
        """
        This implementation is a thin wrapper around `shap.TreeExplainer
        <https://shap-lrjball.readthedocs.io/en/docs_update/generated/shap.TreeExplainer.html>`
        
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
        self.feature_names = self.get_feature_names(self.X)
        self.number_of_features = self.get_number_of_features(number_of_features)
        self.kwargs = kwargs
        self.sample_index: int = None

        natural_language_text_empty = (
            "The {} features which contributed most to the prediction of this"
            " particular sample were: {}."
        )
        method_text_empty = (
            "The feature importance was calculated using the SHAP method."
        )
        sentence_text_empty = "'{}' ({:.2f})"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )

        self.logger = self.setup_logger(self.explanation_name)

        self._calculate_importance()

    def _calculate_importance(self) -> None:
        """Explain model predictions using SHAP library

        Returns:
            None.
        """
        self.explainer = shap.TreeExplainer(self.model, **self.kwargs)
        self.shap_values = self.explainer.shap_values(self.X)

        # if isinstance(self.explainer.expected_value, np.ndarray):
        #     self.explainer.expected_value = self.explainer.expected_value[0]
        # assert isinstance(
        #     self.explainer.expected_value, float
        # ), "self.explainer.expected_value has wrong type"

    def get_feature_values(self, sample_index: int = 0) -> List[Tuple[str, float]]:
        """Extract the feature name and its importance per sample
        - get absolute values to get the strongst postive and negative contribution
        - sort by importance -> highst to lowest


        Args:
            sample_index (int, optional): sample for which the explanation should
            be returned. Defaults to 0.

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        if not self.is_classifier:
            indexes = np.argsort(abs(self.shap_values[sample_index, :]))
            sample_shap_value = self.shap_values
        else:
            indexes = np.argsort(
                abs(self.shap_values[self.prediction][sample_index, :])
            )
            sample_shap_value = self.shap_values[self.prediction]
            self.logger.info(
                f"SHAP values are taken from predicted class '{self.prediction}'"
            )

        feature_values = []
        for index in indexes.tolist()[::-1]:
            feature_values.append((
                self.feature_names[index],
                sample_shap_value[sample_index, index],
            ))
        return feature_values

    def plot(self, sample_index: int, kind: str = "bar") -> None:
        """Plot the shap values

        Args:
            sample_index (int, optional): DESCRIPTION. Defaults to 0.
            kind (TYPE, optional): DESCRIPTION. Defaults to "bar".

        Returns:
            None:
        """
        if sample_index != self.sample_index:
            raise ValueError(
                "the provided index sample does not match the index the importance is"
                " calculated for. re-run .explain(sample_index) to plot the correct"
                " sample"
            )
        if kind == "bar":
            self.fig = self._bar_plot(sample_index)
        elif kind == "shap":
            self.fig = self._shap_plot(sample_index)
        else:
            raise Exception(f'Value of "kind = {kind}" is not supported!')

    def _bar_plot(self, sample_index: int) -> plt.Figure:
        """Create a bar plot of the shape values for a selected sample

        Args:
            sample_index (int, optional): sample for which the explanation should
                be returned. Defaults to 0.

        Returns:
            plt.figure

        """
        if not self.is_classifier:
            shap_value = self.shap_values
        else:
            shap_value = self.shap_values[self.prediction]

        indexes = np.argsort(abs(shap_value[sample_index, :]))
        sorted_idx = indexes.tolist()[::-1][: self.number_of_features]

        width = shap_value[sample_index, sorted_idx]
        labels = [self.feature_names[i] for i in sorted_idx]
        y = np.arange(self.number_of_features, 0, -1)

        fig = plt.figure(figsize=(6, max(2, int(0.5 * self.number_of_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel("Shap Values")
        plt.tight_layout()
        plt.show()
        return fig

    def _shap_plot(self, sample_index: int) -> plt.Figure:
        """Visualize the first prediction's explanation

        Args:
            sample_index (int, optional): sample for which the explanation should
                be returned. Defaults to 0.

        Returns:
            plt.figure: return a matplotlib figure containg the plot
        """
        if not self.is_classifier:
            base_value = self.explainer.expected_value
            shap_value = np.around(self.shap_values[sample_index, :], decimals=2)
        else:
            base_value = self.explainer.expected_value[self.prediction]
            shap_value = np.around(
                self.shap_values[self.prediction][sample_index, :], decimals=2
            )

        shap.force_plot(
            base_value=base_value,
            shap_values=shap_value,
            features=self.X.iloc[sample_index, :],
            matplotlib=True,
            show=False,
        )

        fig = plt.gcf()
        fig.set_figheight(4)
        fig.set_figwidth(8)
        plt.show()
        return fig

    def _log_output(self, sample_index: int) -> None:
        """Log the prediction values of the sample

        Args:
            sample (int): DESCRIPTION.

        Returns:
            None.
        """
        if not self.is_classifier:
            message = f"The expected_value was: {self.explainer.expected_value}"
        else:
            message = (
                "The expected_value was:"
                f" {self.explainer.expected_value[self.prediction]}"
            )

        self.logger.debug(message)
        self.logger.debug(f"The y_value was: {self.y.values[sample_index]}")
        self.logger.debug(f"The predicted value was: {self.prediction}")

    def _setup(self, sample_index: int, sample_name: str) -> None:
        """Helper function to call all methods to create the explanations

        Args:
            sample_index (TYPE): DESCRIPTION.
            sample_name (TYPE): DESCRIPTION.

        Returns:
            None.
        """
        self.sample_index = sample_index
        self._log_output(sample_index)
        self.feature_values = self.get_feature_values(sample_index)
        self.sentences = self.get_sentences()
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()
        self.plot_name = self.get_plot_name(sample_name)

    def explain(
        self,
        sample_index: int,
        sample_name: Optional[str] = None,
        separator: str = "\n",
    ) -> Explanation:
        """Main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample_index (int): number of the sample to create the explanation for

        Returns:
            Explanation: Explanation object containg the explainations
        """
        sample_name = self.get_sample_name(sample_index, sample_name)
        self.prediction = self.get_prediction(sample_index)
        self.score_text = self.get_score_text()

        self._setup(sample_index, sample_name)
        self.explanation = Explanation(
            self.score_text, self.method_text, self.natural_language_text, separator
        )
        return self.explanation
