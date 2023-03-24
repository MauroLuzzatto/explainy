from explainy.explanations.permutation_explanation import PermutationExplanation

from .utils import get_regression_model


def test_permuation_explanation_4_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 4
    sample_index = 1

    explainer = PermutationExplanation(X_test, y_test, model, number_of_features)

    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the Permutation Feature"
        " Importance method."
    )
    assert (
        explanation.natural_language_text
        == "The four features which were most important for the predictions"
        " were: 'bmi' (0.15), 's5' (0.12), 'bp' (0.04), and 'age' (0.02)."
    )


def test_permuation_explanation_8_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 8
    sample_index = 1

    explainer = PermutationExplanation(X_test, y_test, model, number_of_features)

    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the Permutation Feature"
        " Importance method."
    )
    assert (
        explanation.natural_language_text
        == "The eight features which were most important for the predictions"
        " were: 'bmi' (0.15), 's5' (0.12), 'bp' (0.04), 'age' (0.02), 's2'"
        " (-0.00), 'sex' (-0.00), 's3' (-0.00), and 's1' (-0.01)."
    )


if __name__ == "__main__":
    test_permuation_explanation_4_features()
    test_permuation_explanation_8_features()
