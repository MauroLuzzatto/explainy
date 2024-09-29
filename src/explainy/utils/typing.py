from dataclasses import dataclass
from typing import Dict, Union

from sklearn.base import ClassifierMixin, RegressorMixin

ModelType = Union[ClassifierMixin, RegressorMixin]

Config = Dict[str, str]


# TODO: create a dataclass config with all the default values
@dataclass
class ConfigClass:
    file_name: str = "explanations.csv"
    folder_name: str = "output"
