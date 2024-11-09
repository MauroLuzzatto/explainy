import textwrap
import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.logger import Logger
from explainy.utils.typing import Config, ModelType
from explainy.utils.utils import num_to_str


class LinearSurrogate(ExplanationBase):
    """Contrastive, global Explanation"""

    explanation_type: str = "global"
    explanation_style: str = "contrastive"
    explanation_name: str = "surrogate"

    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, np.ndarray],
        model: ModelType,
        number_of_features: int = 4,
        config: Optional[Config] = None,
        **kwargs: dict,
    ):
        super(LinearSurrogate, self).__init__(model, config)
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
        self.number_of_features = number_of_features
        self.kwargs = kwargs

        self.feature_names = self.get_feature_names(self.X)
        self.sample_index: Optional[int] = None

        natural_language_text_empty = (
            "The features which were most important for the predictions were: {}."
        )

        method_text_empty = (
            "The feature importance was calculated using a {} surrogate model."
        )

        sentence_text_empty = "'{}' ({:.2f})"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )
        self.logger = Logger(self.explanation_name, self.path_log).get_logger()
        self._calculate_importance()

    def _calculate_importance(self) -> None:
        """Train a surrogate model on the predicted values from the original model"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            y_hat = self.model.predict(self.X.values)

        self.surrogate_model = self.get_surrogate_model()
        self.surrogate_model.fit(self.X, y_hat)

        # get absolute coefficients
        self.coefficients = abs(self.surrogate_model.coef_)
        self.logger.info(f"Coefficients: {self.coefficients.shape}")

        score_docstring: str = self.surrogate_model.score.__doc__.strip()
        score_description: str = (
            score_docstring.split("\n")[0].strip().lower()[:-1].split("return the ")[-1]
        )
        self.logger.info(
            f"Surrogate Model score ({score_description}):"
            f" {self.surrogate_model.score(self.X, y_hat):.2f}"
        )

    def get_surrogate_model(self) -> ModelType:
        """Get the surrogate model based on the kind of model"""

        if self.is_classifier:
            estimator = LogisticRegression
        else:
            estimator = LinearRegression

        return estimator(**self.kwargs)

    def importance(self) -> str:
        """Return the importance of the surrogate model

        Returns:
            str: importance of the surrogate model

        """
        return super().importance()

    def plot(self, sample_index: int) -> None:
        """Plot the sorted permutation feature importance using a barplot

        Returns:
            plt.figure: a figure object
        """
        sorted_idx = self.feature_importance.argsort()
        # wrap labels longer than 40 characters
        labels = [textwrap.fill(self.feature_names[i], width=40) for i in sorted_idx][
            -self.number_of_features :
        ]

        width = [self.feature_importance[i] for i in sorted_idx][
            -self.number_of_features :
        ]
        y = np.arange(self.number_of_features)

        self.fig = plt.figure(figsize=(6, max(2, int(0.6 * self.number_of_features))))
        plt.barh(y=y, width=width, height=0.5)
        plt.yticks(y, labels)
        plt.xlabel(
            f"Linear Surrogate Model Feature Importance for class {self.prediction}"
        )
        plt.tight_layout()
        plt.show()

    def get_method_text(self) -> str:
        """Define the method introduction text of the explanation type.

        Returns:
            str: method_text explanation
        """
        method_text = self.method_text_empty.format(
            self.surrogate_model.__class__.__name__,
            num_to_str[self.number_of_features].capitalize(),
        )
        return method_text

    def get_natural_language_text(self) -> str:
        """Define the natural language output using the feature names and its
        values for this explanation type

        Returns:
            str: natural_language_text explanation
        """
        sentences = super().get_sentences()
        return self.natural_language_text_empty.format(sentences)

    def get_feature_values(self) -> None:
        """Get the feature values sorted by importance for linear surrogate models"""

        if self.is_classifier:
            # explain the predicted class using the feature importance for the predicted class
            target_class_index = list(self.surrogate_model.classes_).index(
                self.prediction
            )
            # Note: we get the coefficients for each class
            self.feature_importance = self.coefficients[target_class_index, :]
        else:
            self.feature_importance = self.coefficients

        self.feature_values = sorted(
            list(zip(self.feature_names, self.feature_importance)),
            key=lambda x: x[1],
            reverse=True,
        )

    def explain(
        self,
        sample_index: int,
        sample_name: Optional[str] = None,
        separator: str = "\n",
    ) -> Explanation:
        """Main function to create the explanation of the given sample.

        The method_text, natural_language_text and the plots are create per sample.

        Args:
            sample_index (int): number of the sample to create the explanation for
            sample_name (str, optional): name of the sample. Defaults to None.
            separator (str, optional): separator for the string concatenation. Defaults to '\n'.

        Returns:
            Explanation: Explanation object
        """
        self.sample_index = sample_index
        sample_name = self.get_sample_name(sample_index, sample_name)
        self.plot_name = self.get_plot_name(sample_name)
        self.prediction = self.get_prediction(sample_index)

        self.get_feature_values()
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()

        self.score_text = self.get_score_text()
        self.explanation = Explanation(
            self.score_text,
            self.method_text,
            self.natural_language_text,
            separator=separator,
        )
        return self.explanation
