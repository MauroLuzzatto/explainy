import os


def create_folder(path: str) -> str:
    """Create folder, if it doesn't already exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def join_text_with_comma_and_and(values: list) -> str:
    """Merge values for text output with commas and only the last value
    with an "and""

    Args:
        values (list): list of values to be merged.

    Returns:
        str: new text.
    """
    if len(values) > 2:
        last_value = values[-1]
        values = ", ".join(values[:-1])
        text = values + ", and " + last_value

    else:
        text = ", and ".join(values)
    return text


def get_number_to_string_dict() -> None:
    """Map number of features to string values"""
    number_text = (
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    )
    num_to_str = {}
    for text, number in zip(number_text, range(1, 21)):
        num_to_str[number] = text
    return num_to_str


num_to_str = get_number_to_string_dict()
