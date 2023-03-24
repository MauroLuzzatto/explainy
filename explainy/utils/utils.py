import os


def create_folder(path):
    """
    create folder, if it doesn't already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_one_hot_sentence(feature_name, feature_value, sentence):
    """
    Create sentence from one-hot-encoded feature value, split the
    column name into feature and value and create sentence
    based on if the value was 1 = True, or 0 = False

    Args:
        feature_name (TYPE): DESCRIPTION.
        feature_value (TYPE): DESCRIPTION.
        sentence (TYPE): DESCRIPTION.

    Returns:
        sentence_filled (TYPE): DESCRIPTION.

    """
    column, value = feature_name.split(" - ")
    if int(feature_value) == 1:
        sentence_filled = sentence.format(column, f"'{value}'")
    else:
        sentence_filled = sentence.format(column, f"not '{value}'")
    return sentence_filled
