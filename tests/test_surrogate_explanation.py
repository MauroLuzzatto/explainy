from explainy.explanations.surrogate_model_explanation import SurrogateModelExplanation

from .utils import get_regression_model


def test_counterfactual_explanation_4_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 4
    sample_index = 1

    explainer = SurrogateModelExplanation(X_test, y_test, model, number_of_features)
    explanation = explainer.explain(sample_index)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using a DecisionTreeRegressor"
        " surrogate model. Four tree nodes are shown."
    )
    # assert (
    #     natural_language_text
    #     == "The features which were most important for the predictions were as follows: The samples got a value of 109.77 if 's5' was less or equal than 0.03, and 'bmi' was less or equal than 0.01. The samples got a value of 166.01 if 's5' was less or equal than 0.03, and 'bmi' was greater than 0.01. The samples got a value of 172.11 if 's5' was greater than 0.03, and 'bmi' was less or equal than 0.01. The samples got a value of 239.20 if 's5' was greater than 0.03, and 'bmi' was greater than 0.01."
    # )


def test_counterfactual_explanation_8_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 8
    sample_index = 1

    explainer = SurrogateModelExplanation(X_test, y_test, model, number_of_features)
    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using a DecisionTreeRegressor"
        " surrogate model. Eight tree nodes are shown."
    )
    # assert (
    #     natural_language_text
    #     == "The features which were most important for the predictions were as follows: The samples got a value of 142.58 if 's5' was less or equal than 0.03, 'bmi' was greater than 0.01, and 'bp' was less or equal than 0.01. The samples got a value of 202.45 if 's5' was less or equal than 0.03, 'bmi' was greater than 0.01, and 'bp' was greater than 0.01. The samples got a value of 98.88 if 's5' was less or equal than 0.03, 'bmi' was less or equal than 0.01, and 's5' was less or equal than -0.01. The samples got a value of 136.99 if 's5' was less or equal than 0.03, 'bmi' was less or equal than 0.01, and 's5' was greater than -0.01. The samples got a value of 248.83 if 's5' was greater than 0.03, 'bmi' was greater than 0.01, and 's1' was less or equal than 0.05. The samples got a value of 208.70 if 's5' was greater than 0.03, 'bmi' was greater than 0.01, and 's1' was greater than 0.05. The samples got a value of 166.06 if 's5' was greater than 0.03, 'bmi' was less or equal than 0.01, and 's6' was less or equal than 0.03. The samples got a value of 208.38 if 's5' was greater than 0.03, 'bmi' was less or equal than 0.01, and 's6' was greater than 0.03."
    # )


if __name__ == "__main__":
    # model, X_test, y_test = get_classification_model()

    test_counterfactual_explanation_4_features()
    test_counterfactual_explanation_8_features()
