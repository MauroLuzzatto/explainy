import os
import shutil
import subprocess
import warnings
from typing import Optional, Union

import graphviz
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.logger import Logger
from explainy.utils.surrogate_plot import GraphvizNotFoundError, SurrogatePlot
from explainy.utils.surrogate_text import SurrogateText
from explainy.utils.typing import Config, ModelType
from explainy.utils.utils import num_to_str


class TreeSurrogate(ExplanationBase):
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
        super(TreeSurrogate, self).__init__(model, config)
        """Init the specific explanation class, the base class is "Explanation"
        """
        self.X = X
        self.y = y
        self.feature_names = self.get_feature_names(self.X)
        self.kwargs = kwargs
        self.sample_index: Optional[int] = None

        self.number_of_features = int(np.log2(number_of_features))
        self.number_of_groups = number_of_features

        natural_language_text_empty = (
            "The following thresholds were important for the predictions: {}"
        )

        method_text_empty = (
            "The feature importance was calculated using a {} surrogate model. {} tree"
            " nodes are shown."
        )

        if self.is_classifier:
            sentence_text_empty = "\nThe sample is assigned class {} if {}"
        else:
            sentence_text_empty = "\nThe sample has a value of {:.2f} if {}"

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
            estimator = DecisionTreeClassifier
        else:
            estimator = DecisionTreeRegressor
        return estimator(max_depth=self.number_of_features, **self.kwargs)

    def importance(self) -> str:
        """Return the importance of the surrogate model

        Returns:
            str: importance of the surrogate model
        """
        tree_rules = export_text(self.surrogate_model, feature_names=self.feature_names)
        return tree_rules

    def get_natural_language_text(self) -> str:
        """Define the natural language output using the feature names and its
        values for this explanation type

        Returns:
            str: natural_language_text explanation
        """
        surrogateText = SurrogateText(
            text=self.sentence_text_empty,
            model=self.surrogate_model,
            X=self.X,
            feature_names=self.feature_names,
        )
        sentences = surrogateText.get_text()
        return self.natural_language_text_empty.format(sentences)

    def get_method_text(self) -> str:
        """Define the method introduction text of the explanation type.

        Returns:
            str: method_text explanation
        """
        method_text = self.method_text_empty.format(
            self.surrogate_model.__class__.__name__,
            num_to_str[self.number_of_groups].capitalize(),
        )
        return method_text

    def plot(self, sample_index: int, precision: int = 2, **kwargs: dict) -> None:
        """Use Graphviz to plot the decision tree"""
        if shutil.which("dot") is None:
            raise GraphvizNotFoundError(
                "Graphviz not found. Please install it following the instructions in"
                " the README."
            )
        surrogatePlot = SurrogatePlot(precision=precision, **kwargs)

        self.dot_file = surrogatePlot(
            model=self.surrogate_model,
            feature_names=self.feature_names,
        )

        name, extension = os.path.splitext(self.plot_name)
        graphviz_source = graphviz.Source(
            self.dot_file,
            filename=os.path.join(self.path_plot, name),
            format=extension.replace(".", ""),
        )
        try:
            # graphviz_source.view()
            display(graphviz_source)
        except subprocess.CalledProcessError:
            warnings.warn("plot already open!")

    def save(self, sample_index: int, sample_name: Optional[str] = None) -> None:
        """Save the explanations to a csv file, save the plots

        Args:
            sample_index (int): index of sample in scope
            sample_name (str): name of the sample in scope

        Returns:
            None.
        """
        if not sample_name:
            sample_name = sample_index

        self.save_csv(sample_name)
        path = os.path.join(self.path_plot, f"{self.plot_name}.dot")
        with open(path, "w") as file:
            file.write(self.dot_file)

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

    def get_feature_values(self):
        pass
