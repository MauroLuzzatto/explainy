"""Global Surrogate Model
----------------------
A global surrogate model is an interpretable model that is trained to approximate the
predictions of a black box model. We can draw conclusions about the black box model
by interpreting the surrogate model [1].

Characteristics
===============
- global
- contrastive

Source
======
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019.
https://christophm.github.io/interpretable-ml-book/
"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.explanations.linear_surrogate import LinearSurrogate
from explainy.explanations.tree_surrogate import TreeSurrogate
from explainy.utils.typing import Config, ModelType

surrogate_types = ["tree", "linear"]


class SurrogateModelExplanation(ExplanationBase):
    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, np.ndarray],
        model: ModelType,
        number_of_features: int = 4,
        config: Optional[Config] = None,
        kind: Literal["tree", "linear"] = "tree",
        **kwargs: dict,
    ):
        super(SurrogateModelExplanation, self).__init__(model, config)
        """Init the specific explanation class, the base class is "Explanation"

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            number_of_features (int, optional): number of features to show. Defaults to 4.
            config (dict, optional): config dictionary. Defaults to None.
            kind (str, optional): kind of surrogate model. Defaults to "tree".
            **kwargs (dict): hyperparamters for the surrogate model

        Returns:
            None.
        """
        self.X = X
        self.y = y
        self.kind = kind
        self.number_of_features = number_of_features
        self.kwargs = kwargs

        self.sample_index: Optional[int] = None

        self.surrogate = self._get_surrogate()
        self.number_of_features = self.surrogate.number_of_features

    def _get_surrogate(self):
        if self.kind == "tree":
            return TreeSurrogate(
                self.X,
                self.y,
                self.model,
                self.number_of_features,
                self.config,
                **self.kwargs,
            )

        elif self.kind == "linear":
            return LinearSurrogate(
                self.X,
                self.y,
                self.model,
                self.number_of_features,
                self.config,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"'{self.kind}' is not a valid option, select from {surrogate_types}"
            )

    def importance(self):
        return self.surrogate.importance()

    def plot(self, *args, **kwargs):
        return self.surrogate.plot(*args, **kwargs)

    def save(self, sample_index: int, sample_name: Optional[str] = None) -> None:
        return self.surrogate.save(sample_index, sample_name)

    def _calculate_importance(self):
        return self.surrogate._calculate_importance()

    def get_feature_values(self):
        return self.surrogate.get_feature_values()

    def explain(
        self,
        sample_index: int,
        sample_name: Optional[str] = None,
        separator: str = "\n",
    ) -> Explanation:
        return self.surrogate.explain(sample_index, sample_name, separator)
