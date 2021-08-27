from src.explanation.ExplanationMixin import ExplanationMixin


def test_join_text_with_comma_and_and():

    mixin = ExplanationMixin()
    text = mixin.join_text_with_comma_and_and(["this", "is", "a", "test"])
    assert "this, is, a, and test" == text


def test_get_number_to_string_dict():

    mixin = ExplanationMixin()
    mixin.get_number_to_string_dict()

    print(mixin.num_to_str)
    assert "two" == mixin.num_to_str[2]
    assert "twenty" == mixin.num_to_str[20]
    assert "five" == mixin.num_to_str[5]
