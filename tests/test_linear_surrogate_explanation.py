from explainy.explanations.surrogate_model_explanation import SurrogateModelExplanation

from .utils import get_regression_model


def test_linear_counterfactual_explanation_4_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 4
    sample_index = 1

    explainer = SurrogateModelExplanation(
        X_test, y_test, model, number_of_features, kind="linear"
    )
    explanation = explainer.explain(sample_index)
    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using a LinearRegression surrogate"
        " model."
    )
    assert (
        explanation.natural_language_text
        == "The features which were most important for the predictions were: 's5'"
        " (919.08), 's1' (872.37), 'bmi' (603.56), and 's2' (591.68)."
    )


def test_linear_counterfactual_explanation_3_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 3
    sample_index = 1

    explainer = SurrogateModelExplanation(
        X_test, y_test, model, number_of_features, kind="linear"
    )
    explanation = explainer.explain(sample_index, separator=None)

    print(explanation)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using a LinearRegression surrogate"
        " model."
    )
    assert (
        explanation.natural_language_text
        == "The features which were most important for the predictions were: 's5'"
        " (919.08), 's1' (872.37), and 'bmi' (603.56)."
    )
