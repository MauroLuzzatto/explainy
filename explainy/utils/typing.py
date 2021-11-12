from typing import Union

from sklearn.base import ClassifierMixin, RegressorMixin

ModelType = Union[ClassifierMixin, RegressorMixin]
