"""Permutation feature importance
------------------------------
Permutation feature importance measures the increase in the prediction error of the model 
after we permuted the feature's values, which breaks the relationship between 
the feature and the true outcome [1].

Permutation importance does not reflect to the intrinsic predictive value of 
a feature by itself but how important this feature is for a particular model [2].

Characteristics
===============
- global
- non-contrastive

Source
======
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. 
https://christophm.github.io/interpretable-ml-book/

[2] https://scikit-learn.org/stable/modules/permutation_importance.html

"""

import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.typing import Config, ModelType


class PermutationExplanation(ExplanationBase):
    """Non-contrastive, global Explanation"""

    explanation_type: str = "global"
    explanation_style: str = "non-contrastive"
    explanation_name: str = "permutation"

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: ModelType,
        number_of_features: int = 4,
        config: Optional[Config] = None,
        n_repeats: Optional[int] = 30,
        random_state: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super(PermutationExplanation, self).__init__(model, config)
        """        
        This implementation is a thin wrapper around `sklearn.inspection.permutation_importance
        <https://scikit-learn.org/stable/modules/permutation_importance.html>`
        
        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            number_of_features (bool): boolean value to generate sparse or non sparse explanation
            config: Dict = None

        Returns:
            None.

        """
        self.X = X
        self.y = y
        self.feature_names = self.get_feature_names(self.X)
        self.number_of_features = self.get_number_of_features(number_of_features)
        self.kwargs = kwargs
        self.kwargs["n_repeats"] = n_repeats
        self.kwargs["random_state"] = random_state

        natural_language_text_empty = (
            "The {} features which were most important for the predictions were: {}."
        )
        method_text_empty = (
            "The feature importance was calculated using the Permutation"
            " Feature Importance method."
        )
        sentence_text_empty = "'{}' ({:.2f})"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )

        self.logger = self.setup_logger(self.explanation_name)

        self._setup()

    def _calculate_importance(self) -> None:
        """Calculate the feature importance using the Permuation Feature Importance

        Args:
            n_repeats (int, optional): sets the number of times a feature
            is randomly shuffled

        Returns:
            None
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.r = permutation_importance(
                self.model, self.X, self.y.values, **self.kwargs
            )

    def get_feature_values(self) -> List[Tuple[str, float]]:
        """Extract the feature name and its importance per sample,
        sort by importance -> highest to lowest

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        feature_values = []
        for index in self.r.importances_mean.argsort()[::-1]:
            feature_values.append(
                (self.feature_names[index], self.r.importances_mean[index])
            )
        return feature_values

    def _box_plot(self) -> plt.figure:
        """Plot the sorted permutation feature importance using a boxplot

        Returns:
            plt.figure: a figure object

        """
        sorted_idx = self.r.importances_mean.argsort()
        values = self.r.importances[sorted_idx].T
        labels = [self.feature_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, max(2, int(0.5 * self.number_of_features))))
        ax.boxplot(
            values[:, -self.number_of_features :],
            vert=False,
            labels=labels[-self.number_of_features :],
        )
        plt.xlabel("Permutation Feature Importance")
        plt.tight_layout()
        plt.show()
        return fig

    def _bar_plot(self) -> plt.figure:
        """Plot the sorted permutation feature importance using a barplot

        Returns:
            plt.figure: a figure object
        """
        sorted_idx = self.r.importances_mean.argsort()
        labels = [self.feature_names[i] for i in sorted_idx][-self.number_of_features :]
        width = [self.r.importances_mean[i] for i in sorted_idx][
            -self.number_of_features :
        ]
        y = np.arange(self.number_of_features)

        fig = plt.figure(figsize=(6, max(2, int(0.5 * self.number_of_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel(
            "Permutation Feature Importance"
        )  # TODO: add the loss e.g. (loss = R2)
        plt.tight_layout()
        plt.show()
        return fig

    def plot(self, sample_index: int = None, kind: str = "bar") -> None:
        """Plot method that calls different kinds of plot types

        Args:
            kind (TYPE, optional): DESCRIPTION. Defaults to 'bar'.

        Returns:
            None.
        """
        if kind == "bar":
            self.fig = self._bar_plot()
        elif kind == "box":
            self.fig = self._box_plot()
        else:
            raise Exception(f'Value of "kind" is not supported: {kind}!')

    def _setup(self) -> None:
        """Since the plots and values are calculate once per trained model,
        the feature importance computatoin is done at the beginning
        when initating the class

        Returns:
            None.
        """
        self._calculate_importance()
        self.feature_values = self.get_feature_values()
        self.sentences = self.get_sentences()
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()
        self.plot_name = self.get_plot_name()

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
        self.explanation = Explanation(
            self.score_text,
            self.method_text,
            self.natural_language_text,
            separator=separator,
        )
        return self.explanation
