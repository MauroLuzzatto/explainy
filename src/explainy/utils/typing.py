from typing import Union, Dict
from sklearn.base import ClassifierMixin, RegressorMixin

ModelType = Union[ClassifierMixin, RegressorMixin]

Config = Dict[str, str]
