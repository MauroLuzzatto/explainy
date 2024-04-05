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

import os
import shutil
import subprocess
import warnings
from typing import Optional, Union

import graphviz
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.surrogate_plot import GraphvizNotFoundError, SurrogatePlot
from explainy.utils.surrogate_text import SurrogateText
from explainy.utils.typing import Config, ModelType
from explainy.utils.utils import num_to_str


class SurrogateModelExplanation(ExplanationBase):
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
        kind: str = "tree",
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
        self.feature_names = self.get_feature_names(self.X)
        self.number_of_features = int(np.log2(number_of_features))
        self.number_of_groups = number_of_features
        self.kind = kind
        self.kwargs = kwargs
        self.sample_index: Optional[int] = None

        kinds = ["tree", "linear"]
        assert (
            self.kind in kinds
        ), f"'{self.kind}' is not a valid option, select from {kinds}"

        # define text defaults
        natural_language_text_empty = (
            "The following thresholds were important for the predictions: {}"
        )
        method_text_empty = (
            "The feature importance was calculated using a {} surrogate model."
            " {} tree nodes are shown."
        )

        if self.is_classifier:
            sentence_text_empty = "\nThe sample is assigned class {} if {}"
        else:
            sentence_text_empty = "\nThe sample has a value of {:.2f} if {}"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )
        self.logger = self.setup_logger(self.explanation_name)

        self._setup()

    def _calculate_importance(self) -> None:
        """Train a surrogate model on the predicted values from the original model

        Raises:
            Exception: if the 'kind' is not known

        """
        if self.kind == "tree" and not self.is_classifier:
            estimator = DecisionTreeRegressor
        elif self.kind == "tree" and self.is_classifier:
            estimator = DecisionTreeClassifier
        elif self.kind == "linear" and not self.is_classifier:
            estimator = LinearRegression
        elif self.kind == "linear" and self.is_classifier:
            estimator = LogisticRegression
        else:
            raise ValueError(f"Value of kind '{self.kind}' is not supported!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            y_hat = self.model.predict(self.X.values)

        self.surrogate_model = self.get_surrogate_model(estimator)
        self.surrogate_model.fit(self.X, y_hat)

        score_docstring: str = self.surrogate_model.score.__doc__.strip()
        score_description: str = (
            score_docstring.split("\n")[0].strip().lower()[:-1].split("return the ")[-1]
        )

        self.logger.info(
            f"Surrogate Model score ({score_description}):"
            f" {self.surrogate_model.score(self.X, y_hat):.2f}"
        )

    def get_surrogate_model(self, estimator: ModelType) -> ModelType:
        """Get the surrogate model per kind with the defined hyperparamters

        Args:
            estimator (ModelType): surrogate estimator

        Returns:
            ModelType: surrogate estimator with hyperparamters

        """
        if self.kind == "tree":
            surrogate_model = estimator(
                max_depth=self.number_of_features, **self.kwargs
            )
        elif self.kind == "linear":
            surrogate_model = estimator(**self.kwargs)
        return surrogate_model

    def get_feature_values(self):
        pass

    def importance(self) -> str:
        """Return the importance of the surrogate model

        Returns:
            str: importance of the surrogate model

        """
        if isinstance(
            self.surrogate_model,
            (DecisionTreeClassifier, DecisionTreeRegressor),
        ):
            tree_rules = export_text(
                self.surrogate_model, feature_names=self.feature_names
            )
        return tree_rules

    def plot(self, sample_index: int = None) -> None:
        """Plot the surrogate model

        Args:
            sample_index (int): index of the sample in scope.

        Raises:
            Exception: if the type of kind is not supported

        """
        if self.kind == "tree":
            self._plot_tree()
        elif self.kind == "linear":
            self._plot_bar()
        else:
            raise Exception(f'Value of "kind" is not supported: {self.kind}!')

    def _plot_bar(self) -> None:
        raise NotImplementedError("to be done")

    def _plot_tree(self, precision: int = 2, **kwargs: dict) -> None:
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

        with open(
            os.path.join(self.path_plot, f"{self.plot_name}.dot"),
            "w",
        ) as file:
            file.write(self.dot_file)

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

    def _setup(self) -> None:
        """Calculate the feature importance and create the text once

        Returns:
            None.

        """
        self._calculate_importance()
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()

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
            separator (str, optional): seprator for the string concatenation. Defaults to '\n'.

        Returns:
            Explanation: Explanation object containg the explainations
        """
        self.sample_index = sample_index
        sample_name = self.get_sample_name(sample_index, sample_name)
        self.plot_name = self.get_plot_name(sample_name)

        self.prediction = self.get_prediction(sample_index)
        self.score_text = self.get_score_text()
        self.explanation = Explanation(
            self.score_text,
            self.method_text,
            self.natural_language_text,
            separator=separator,
        )
        return self.explanation
