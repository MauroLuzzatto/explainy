"""
Created on Tue Nov 24 21:41:40 2020

@author: mauro

Permutation importance does not reflect to the intrinsic predictive value of 
a feature by itself but how important this feature is for a particular model.

Source:
https://scikit-learn.org/stable/modules/permutation_importance.html

"""
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from src.explanation.ExplanationBase import ExplanationBase



class PermutationExplanation(ExplanationBase):
    """
    Non-contrastive, global Explanation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: sklearn.base.BaseEstimator,
        number_of_features: int,
        config: Dict = None
    ):
        super(PermutationExplanation, self).__init__(
            config
        )
        """
        Init the specific explanation class, the base class is "Explanation"

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
        self.model = model
        self.feature_names = list(self.X)
        self.number_of_features = self.get_number_of_features(number_of_features)

        natural_language_text_empty = (
            "The {} features which were most important for the predictions were (from highest to lowest): {}."
        )
        method_text_empty = (
            "The feature importance was calculated using the Permutation Feature Importance method."
        )
        sentence_text_empty = "'{}' ({:.2f})"

        self.natural_language_text_empty = self.config.get("natural_language_text_empty", natural_language_text_empty)
        self.method_text_empty = self.config.get("method_text_empty", method_text_empty)
        self.sentence_text_empty = self.config.get("sentence_text_empty", sentence_text_empty)

        self.explanation_name = "permutation"
        self.logger = self.setup_logger(self.explanation_name)
        
        self._setup()
        
    
    def _calculate_explanation(self, n_repeats=30):
        """
        conduct the Permutation Feature Importance and get the importance

         Args:
            n_repeats (int, optional): sets the number of times a feature
            is randomly shuffled

        Returns:
            None
        """
        self.r = permutation_importance(
            self.model,
            self.X.values,
            self.y.values,
            n_repeats=n_repeats,
            random_state=0,
        )

    def get_feature_values(self):
        """
        extract the feature name and its importance per sample

        Args:
            sample (int, optional): sample for which the explanation should
            be returned. Defaults to 0.

        Returns:
            feature_values (list(tuple(str, float))): list of tuples for each
            feature and its importance of a sample.

        """
        feature_values = []
        # sort by importance -> highst to lowest
        for index in self.r.importances_mean.argsort()[::-1][
            : self.number_of_features
        ]:
            feature_values.append(
                (self.feature_names[index], self.r.importances_mean[index])
            )
        return feature_values

    def box_plot(self):
        """
        plot the sorted permutation feature importance using a boxplot

        Returns:
            None.

        """
        sorted_idx = self.r.importances_mean.argsort()
        values = self.r.importances[sorted_idx].T
        labels = [self.feature_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(
            figsize=(6, max(2, int(0.5 * self.number_of_features)))
        )
        ax.boxplot(
            values[:, -self.number_of_features :],
            vert=False,
            labels=labels[-self.number_of_features :],
        )
        plt.xlabel("Permutation Feature Importance")
        plt.tight_layout()
        plt.grid()
        plt.show()
        return fig
        
    def bar_plot(self):
        """
        Bar plot of the feature importance

        Returns:
            None.

        """
        sorted_idx = self.r.importances_mean.argsort()
        values = self.r.importances[sorted_idx].T
        labels = [self.feature_names[i] for i in sorted_idx][
            -self.number_of_features :
        ]

        width = np.median(values[:, -self.number_of_features :], axis=0)
        y = np.arange(self.number_of_features)

        fig = plt.figure(figsize=(6, max(2, int(0.5 * self.number_of_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel("Permutation Feature Importance")  # TODO: add the loss e.g. (loss = R2)
        plt.tight_layout()
        plt.grid()
        plt.show()
        return fig

    def plot(self, kind='bar'):
        """
        

        Args:
            kind (TYPE, optional): DESCRIPTION. Defaults to 'bar'.

        Returns:
            None.

        """
        if kind == 'bar':
            self.fig = self.bar_plot()
        elif kind == 'box':
            self.fig = self.box_plot()
        else:
            raise

    def _setup(self):
        """
        Since the plots and values are calculate once per trained model,
        the feature importance computatoin is done at the beginning
        when initating the class

        Returns:
            None.
        """
        self._calculate_explanation()
        self.feature_values = self.get_feature_values()
        
        self.sentences = self.get_sentences(self.feature_values, self.sentence_text_empty)
        self.natural_language_text = self.get_natural_language_text(
   )
        self.method_text = self.get_method_text()        
        self.plot_name = self.get_plot_name()
        
    def save(self, sample_name):
        """
        

        Args:
            sample_name (TYPE): DESCRIPTION.

        Returns:
            None.

        """
        self.save_csv(sample_name)
                
        self.fig.savefig(
            os.path.join(self.path_plot, self.plot_name),
            bbox_inches="tight",
        )

    def explain(self, sample_index, sample_name=None, separator='\n'):
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
        self.score_text = self.get_score_text()
        self.explanation =  self.get_explanation(separator)
        return self.explanation
        