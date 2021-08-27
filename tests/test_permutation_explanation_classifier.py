# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.explanation.permutation_explanation import PermutationExplanation


def get_classification_model():

    iris = load_iris()
    
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=iris.feature_names)
    y_test = pd.DataFrame(y_test)
    
    model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    return model, X_test, y_test


def test_permuation_explanation_2_features():

    number_of_features = 2
    sample_index = 1

    explainer = PermutationExplanation(
        X_test, y_test, model, number_of_features
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    
    score_text, method_text, natural_language_text = explanation

    assert (
        score_text
        == "The RandomForestClassifier used 4 features to produce the predictions. The prediction of this sample was 1.0."
    )
    assert (
        method_text
        == "The feature importance was calculated using the Permutation Feature Importance method."
    )
    assert (
        natural_language_text
        == "The two features which were most important for the predictions were (from highest to lowest): 'petal length (cm)' (0.16), and 'petal width (cm)' (0.16)."
    )

def test_permuation_explanation_4_features():

    number_of_features = 4
    sample_index = 5

    explainer = PermutationExplanation(
        X_test, y_test, model, number_of_features
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    score_text, method_text, natural_language_text = explanation

    assert (
        score_text
        == "The RandomForestClassifier used 4 features to produce the predictions. The prediction of this sample was 1.0."
    )
    assert (
        method_text
        == "The feature importance was calculated using the Permutation Feature Importance method."
    )
    assert (
        natural_language_text
        == "The four features which were most important for the predictions were (from highest to lowest): 'petal length (cm)' (0.16), 'petal width (cm)' (0.16), 'sepal width (cm)' (0.00), and 'sepal length (cm)' (0.00)."
    )



if __name__ == "__main__":
    
    model, X_test, y_test = get_classification_model()
    
    test_permuation_explanation_2_features()
    test_permuation_explanation_4_features()
    
    