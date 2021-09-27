# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

from explainy.explanations.counterfactual_explanation import \
    CounterfactualExplanation

from .utils import get_classification_model


def test_counterfactual_explanation_4_features():

    model, X_test, y_test = get_classification_model()

    number_of_features = 4
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test, y_test, model, number_of_features, y_desired=2
    )

    explanation = explainer.explain(sample_index, separator=None)
    score_text, method_text, natural_language_text = explanation

    print(score_text)
    print(method_text)
    print(natural_language_text)

    assert (
        score_text
        == "The RandomForestClassifier used 4 features to produce the predictions. The prediction of this sample was 1.0."
    )
    assert (
        method_text
        == "The feature importance is shown using a counterfactual example."
    )
    assert (
        natural_language_text
        == "The sample would have had the desired prediction, if the 'petal width (cm)' was '1.76', the 'petal length (cm)' was '4.0', the 'sepal width (cm)' was '2.85', and the 'sepal length (cm)' was '6.0'."
    )


def test_counterfactual_explanation_8_features():

    model, X_test, y_test = get_classification_model()

    number_of_features = 8
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test, y_test, model, number_of_features, y_desired=2
    )

    explanation = explainer.explain(sample_index, separator=None)
    score_text, method_text, natural_language_text = explanation
    print(score_text)
    print(method_text)
    print(natural_language_text)

    assert (
        score_text
        == "The RandomForestClassifier used 4 features to produce the predictions. The prediction of this sample was 1.0."
    )
    assert (
        method_text
        == "The feature importance is shown using a counterfactual example."
    )
    assert (
        natural_language_text
        == "The sample would have had the desired prediction, if the 'petal width (cm)' was '1.76', the 'petal length (cm)' was '4.0', the 'sepal width (cm)' was '2.85', and the 'sepal length (cm)' was '6.0'."
    )


if __name__ == "__main__":
    test_counterfactual_explanation_4_features()
    test_counterfactual_explanation_8_features()