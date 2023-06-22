from explainy.explanations.permutation_explanation import PermutationExplanation

from .utils import get_classification_model


def test_permuation_explanation_2_features():
    model, X_test, y_test = get_classification_model()

    number_of_features = 2
    sample_index = 1

    explainer = PermutationExplanation(X_test, y_test, model, number_of_features)

    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestClassifier used 4 features to produce the"
        " predictions. The class of this sample was 1."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the Permutation Feature"
        " Importance method."
    )
    assert (
        explanation.natural_language_text
        == "The two features which were most important for the predictions"
        " were: 'petal length (cm)' (0.16), and 'petal width (cm)' (0.16)."
    )


def test_permuation_explanation_4_features():
    model, X_test, y_test = get_classification_model()

    number_of_features = 4
    sample_index = 5

    explainer = PermutationExplanation(X_test, y_test, model, number_of_features)

    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestClassifier used 4 features to produce the"
        " predictions. The class of this sample was 2."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the Permutation Feature"
        " Importance method."
    )
    assert (
        explanation.natural_language_text
        == "The four features which were most important for the predictions"
        " were: 'petal length (cm)' (0.16), 'petal width (cm)' (0.16),"
        " 'sepal width (cm)' (0.00), and 'sepal length (cm)' (0.00)."
    )


if __name__ == "__main__":
    test_permuation_explanation_2_features()
    test_permuation_explanation_4_features()
