import pytest

from explainy.explanations.counterfactual_explanation import CounterfactualExplanation
from explainy.utils.utils import NonConvergenceError

from .utils import get_classification_model


def test_counterfactual_explanation_4_features():
    model, X_test, y_test = get_classification_model()

    number_of_features = 4
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test,
        y_test,
        model,
        y_desired=2,
        number_of_features=number_of_features,
    )
    explanation = explainer.explain(sample_index, separator=None)

    print(explanation.natural_language_text)

    assert (
        explanation.score_text
        == "The RandomForestClassifier used 4 features to produce the"
        " predictions. The class of this sample was 1."
    )
    assert (
        explanation.method_text
        == "The feature importance is shown using a counterfactual example."
    )
    assert (
        explanation.natural_language_text
        == "The sample would have had the desired prediction of '2', if the"
        " 'petal width (cm)' was '1.75', the 'petal length (cm)' was '4.0',"
        " the 'sepal width (cm)' was '3.1', and the 'sepal length (cm)' was"
        " '6.33'."
    )


def test_counterfactual_explanation_8_features():
    model, X_test, y_test = get_classification_model()

    number_of_features = 8
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test,
        y_test,
        model,
        y_desired=2,
        number_of_features=number_of_features,
    )
    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestClassifier used 4 features to produce the"
        " predictions. The class of this sample was 1."
    )
    assert (
        explanation.method_text
        == "The feature importance is shown using a counterfactual example."
    )

    print(explanation.natural_language_text)

    assert (
        explanation.natural_language_text
        == "The sample would have had the desired prediction of '2', if the"
        " 'petal width (cm)' was '1.75', the 'petal length (cm)' was '4.0',"
        " the 'sepal width (cm)' was '3.1', and the 'sepal length (cm)' was"
        " '6.33'."
    )


# def test_counterfactual_explanation__non_convergence_error():
#     model, X_test, y_test = get_classification_model()

#     number_of_features = 2
#     sample_index = 1

#     explainer = CounterfactualExplanation(
#         X_test,
#         y_test,
#         model,
#         y_desired=4,  # this will not converge
#         number_of_features=number_of_features,
#     )

#     with pytest.raises(NonConvergenceError) as exc_info:
#         explainer.explain(sample_index)

#     assert (
#         str(exc_info.value)
#         == "No counterfactual value found, try to decrease the 'delta' value or adjust the desired prediction 'y_desired'"
#     )


def test_counterfactual_explanation__same_y_desired_as_prediction():
    model, X_test, y_test = get_classification_model()

    number_of_features = 4
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test,
        y_test,
        model,
        y_desired=1,  # this is the same value as the prediction
        number_of_features=number_of_features,
    )

    with pytest.warns(Warning):
        explainer.explain(sample_index)


def test_counterfactual_explanation__zero_class_value():
    model, X_test, y_test = get_classification_model()

    number_of_features = 4
    sample_index = 1

    explainer = CounterfactualExplanation(
        X_test,
        y_test,
        model,
        y_desired=0,
        number_of_features=number_of_features,
    )

    explainer.explain(sample_index)

    assert True


if __name__ == "__main__":
    test_counterfactual_explanation_4_features()
    test_counterfactual_explanation_8_features()
