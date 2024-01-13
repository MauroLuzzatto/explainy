from unittest import mock

from explainy.explanations.shap_explanation import ShapExplanation

from tests.utils import get_regression_model


def test_shap_explanation_4_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 4
    sample_index = 1

    explainer = ShapExplanation(X_test, y_test, model, number_of_features)
    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the SHAP method."
    )
    assert (
        explanation.natural_language_text
        == "The four features which contributed most to the prediction of this"
        " particular sample were: 'bmi' (49.60), 's5' (41.64), 'bp' (9.31),"
        " and 's6' (-4.04)."
    )


def test_shap_explanation_8_features():
    model, X_test, y_test = get_regression_model()

    number_of_features = 8
    sample_index = 1

    explainer = ShapExplanation(X_test, y_test, model, number_of_features)
    explanation = explainer.explain(sample_index, separator=None)

    assert (
        explanation.score_text
        == "The RandomForestRegressor used 10 features to produce the"
        " predictions. The prediction of this sample was 251.6."
    )
    assert (
        explanation.method_text
        == "The feature importance was calculated using the SHAP method."
    )
    assert (
        explanation.natural_language_text
        == "The eight features which contributed most to the prediction of this"
        " particular sample were: 'bmi' (49.60), 's5' (41.64), 'bp' (9.31),"
        " 's6' (-4.04), 'age' (-2.47), 's3' (2.25), 's4' (2.10), and 's2'"
        " (0.92)."
    )


def explainer_wrapper():
    model, X_test, y_test = get_regression_model()
    number_of_features = 8
    return ShapExplanation(X_test, y_test, model, number_of_features)


@mock.patch("explainy.explanations.shap_explanation.plt")
def test_shap_plot_bar(mock_plt):
    explainer = explainer_wrapper()
    sample_index = 1
    explainer.explain(sample_index, separator=None)
    explainer.plot(sample_index, kind="bar")

    mock_plt.xlabel.assert_called_once_with("Shap Values")
    # Assert plt.figure got called
    assert mock_plt.figure.called


@mock.patch("explainy.explanations.shap_explanation.plt")
def test_shap_plot_shap(mock_plt):
    explainer = explainer_wrapper()
    sample_index = 1
    explainer.explain(sample_index, separator=None)
    explainer.plot(sample_index, kind="shap")

    mock_plt.gcf().set_figheight.assert_called_once_with(4)
    mock_plt.gcf().set_figwidth.assert_called_once_with(8)
    assert mock_plt.gcf.called


if __name__ == "__main__":
    test_shap_plot_shap()
    test_shap_plot_bar()

    # test_shap_explanation_4_features()
    # test_shap_explanation_8_features()
