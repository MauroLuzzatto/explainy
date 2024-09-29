import matplotlib.pyplot as plt
import pytest

from explainy.core.explanation_base import ExplanationBase


class ExplanationInstance(ExplanationBase):
    def _calculate_importance(self):
        pass

    def plot(self):
        pass

    def get_feature_values(self):
        pass


config = {}


def get_explanation_concrete():
    return ExplanationInstance(config)


def test_get_natural_language_text():
    explainer = get_explanation_concrete()
    explainer.natural_language_text_empty = "{} and {}"
    explainer.number_of_features = 3
    explainer.sentences = "Test sentence"

    natural_language_text = explainer.get_natural_language_text()
    expected = "three and Test sentence"
    print(natural_language_text)
    assert expected == natural_language_text


def test_get_sentences():
    explainer = get_explanation_concrete()
    explainer.feature_values = [("test", 1), ("test2", 2)]
    explainer.number_of_features = 2
    explainer.sentence_text_empty = "feature: {} - value: {}"

    sentences = explainer.get_sentences()
    expected = "feature: test - value: 1, and feature: test2 - value: 2"
    print(sentences)
    assert expected == sentences


def test_get_plot_name():
    explainer = get_explanation_concrete()
    explainer.explanation_name = "testing"
    explainer.number_of_features = 4

    expected = "testing_features_4.png"
    plot_name = explainer.get_plot_name()
    assert expected == plot_name

    expected = "testing_features_4_sample_one.png"
    plot_name = explainer.get_plot_name("one")
    assert expected == plot_name


def test_save__assertion_error():
    explainer = get_explanation_concrete()

    with pytest.raises(AssertionError) as exc_info:
        explainer.save(sample_index=0, sample_name="one")

    assert str(exc_info.value) == "missing the figure object, call `plot()` first"


def test_save_csv__assertion_error():
    explainer = get_explanation_concrete()

    # add a fake figure object
    explainer.fig = "figure-object"
    with pytest.raises(AssertionError) as exc_info:
        explainer.save(sample_index=0, sample_name="one")

    assert (
        str(exc_info.value)
        == "missing the `natural_language_text`, call `explain()` first"
    )


def test_save():
    explainer = get_explanation_concrete()

    # add a fake figure object
    explainer.fig = plt.figure()
    explainer.natural_language_text = "natural_language_text"
    explainer.prediction = 0
    explainer.score_text = "score text"
    explainer.method_text = "method text"
    explainer.plot_name = "plot_name"
    explainer.number_of_features = 3

    output = explainer.save(sample_index=0, sample_name="one")
    assert output is None


if __name__ == "__main__":
    # pytest.main()
    test_get_natural_language_text()
    test_get_sentences()
    test_get_plot_name()
