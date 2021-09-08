# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from explainy.explanation.shap_explanation import ShapExplanation


def get_regression_model():

    diabetes = load_diabetes()
    
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=diabetes.feature_names)
    y_test = pd.DataFrame(y_test)
    
    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)    
    return model, X_test, y_test


def test_shap_explanation_4_features():


    model, X_test, y_test = get_regression_model()

    number_of_features = 4
    sample_index = 1

    explainer = ShapExplanation(
        X_test, y_test, model, number_of_features
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    score_text, method_text, natural_language_text = explanation
    
    print(score_text)
    print(method_text)
    print(natural_language_text)

    assert (
        score_text == 'The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8.'
    )
    assert (
        method_text
        == 'The feature importance was calculated using the SHAP method.'
    )
    assert (
        natural_language_text
        == "The four features which were most important for this particular sample were (from highest to lowest): 'bmi' (49.63), 's5' (41.66), 'bp' (9.40), and 's6' (-4.04)."
    )

def test_shap_explanation_8_features():

    number_of_features = 8
    sample_index = 1

    model, X_test, y_test = get_regression_model()


    explainer = ShapExplanation(
        X_test, y_test, model, number_of_features
    )

    explanation = explainer.explain(
        sample_index, separator=None
    )
    score_text, method_text, natural_language_text = explanation
    print(score_text)
    print(method_text)
    print(natural_language_text)
    

    assert (
        score_text
        == "The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8."
    )
    assert (
        method_text
        == "The feature importance was calculated using the SHAP method."
    )
    assert (
        natural_language_text
        == "The eight features which were most important for this particular sample were (from highest to lowest): 'bmi' (49.63), 's5' (41.66), 'bp' (9.40), 's6' (-4.04), 'age' (-2.41), 's3' (2.25), 's4' (2.10), and 's2' (0.93)."
    )



if __name__ == "__main__":
    
    model, X_test, y_test = get_regression_model()
    
    test_shap_explanation_4_features()
    test_shap_explanation_8_features()
    
    