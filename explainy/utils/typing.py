from typing import Union

from sklearn.base import ClassifierMixin 
from sklearn.base import RegressorMixin

ModelType = Union[ClassifierMixin, RegressorMixin]
