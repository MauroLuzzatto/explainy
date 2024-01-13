from typing import Dict, Union

from sklearn.base import ClassifierMixin, RegressorMixin

ModelType = Union[ClassifierMixin, RegressorMixin]

Config = Dict[str, str]
