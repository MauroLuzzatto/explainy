# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:15:30 2020

@author: mauro
"""

import csv
import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.base import is_classifier

from explainy.core.explanation_mixin import ExplanationMixin
from explainy.logger import Logger
from explainy.utils.typing import ModelType
from explainy.utils.utils import create_folder


class ExplanationBase(ABC, ExplanationMixin):
    def __init__(
        self,
        model: ModelType,
        config: Dict = None,
    ) -> None:
        """Initialize the explanation base class

        Args:
            model (ModelType): trained model that should be explained
            config (Dict, optional): config file that contains explanation settings. Defaults to None.

        """
        self.model = model

        if not config:
            self.config = {}
        else:
            self.config = config

        self.is_classifier = is_classifier(self.model)

        self.folder = self.config.get("folder", "explanation")
        self.file_name = self.config.get("file_name", "explanations.csv")

        self.set_paths()
        self.get_number_to_string_dict()

        if self.is_classifier:
            score_text_empty = (
                "The {} used {} features to produce the predictions. The class"
                " of this sample was {:.0f}."
            )
        else:
            score_text_empty = (
                "The {} used {} features to produce the predictions. The"
                " prediction of this sample was {:.1f}."
            )

        description_text_empty = (
            "This is a {} explanation, it creates {} and {} explanations."
        )
        self.description_text_empty = self.config.get(
            "description_text_empty", description_text_empty
        )
        self.score_text_empty = self.config.get("score_text_empty", score_text_empty)

    def define_explanation_placeholder(
        self,
        natural_language_text_empty: str,
        method_text_empty: str,
        sentence_text_empty: str,
    ) -> None:
        """Set the explanation text, if defined else load it from defaults

        Args:
            natural_language_text_empty (str): natural language explanation placeholder
            method_text_empty (str): method placeholder
            sentence_text_empty (str): sentence text placeholder

        """
        self.natural_language_text_empty = self.config.get(
            "natural_language_text_empty", natural_language_text_empty
        )
        self.method_text_empty = self.config.get("method_text_empty", method_text_empty)
        self.sentence_text_empty = self.config.get(
            "sentence_text_empty", sentence_text_empty
        )

    def get_number_of_features(self, number_of_features: int) -> int:
        """Set the number of features based on the defined number and the max
        number of features

        Args:
            number_of_features (int): number_of_features as input

        Returns:
            int: number_of_features considering the max number of dataset features
        """
        if number_of_features > self.X.shape[1]:
            warnings.warn(
                'The "number_of_features" is larger than the number of dataset'
                f" features. The value is set to {self.X.shape[1]}"
            )
        return min(number_of_features, self.X.shape[1])

    def get_feature_names(self, X: Union[pd.DataFrame, np.array]) -> List[str]:
        """Get the feature names based on the given dataset

        Args:
            X (Union[pd.DataFrame, np.array]): features dataset

        Returns:
            List[str]: list of feature names

        """
        if isinstance(X, pd.DataFrame):
            feature_names = list(X)
        else:
            feature_names = [f"feature_{index}" for index in range(X.shape[1])]
        return feature_names

    def set_paths(self) -> None:
        """Set the paths where the output should be saved

        Returns:
            None.

        """
        self.path = os.path.join(os.path.dirname(os.getcwd()), "reports", self.folder)
        self.path_plot = create_folder(os.path.join(self.path, "plot"))
        self.path_result = create_folder(os.path.join(self.path, "results"))
        self.path_log = create_folder(os.path.join(self.path, "logs"))

    def setup_logger(self, logger_name: str) -> object:
        logger = Logger(logger_name, self.path_log)
        return logger.get_logger()

    @abstractmethod
    def _calculate_importance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def plot(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_feature_values(self):
        raise NotImplementedError("Subclasses should implement this!")

    def importance(self):
        return pd.DataFrame(
            self.feature_values, columns=["Feature", "Importance"]
        ).round(2)

    def get_prediction(self, sample_index: int) -> float:
        """Get the model prediction

        Args:
            sample_index (int): sample_index for a which a predction shall be made

        Returns:
            float: predction of the model for that sample

        """
        return self.model.predict(self.X.values)[sample_index]

    def get_method_text(self) -> None:
        """Generate the output of the method explanation.

        Returns:
            None

        """
        return self.method_text_empty.format(self.num_to_str[self.number_of_features])

    def get_sentences(self) -> None:
        """Generate the output sentences

        Returns:
            None

        """
        values = []
        for feature_name, feature_value in self.feature_values[
            : self.number_of_features
        ]:
            values.append(self.sentence_text_empty.format(feature_name, feature_value))
        sentences = self.join_text_with_comma_and_and(values)
        return sentences

    def get_natural_language_text(self) -> str:
        """Generate the output of the explanation in natural language.

        Returns:
            str: return the natural_language_text explanation

        """
        return self.natural_language_text_empty.format(
            self.num_to_str[self.number_of_features], self.sentences
        )

    def get_description_text(self) -> str:
        """WIP

        Example:
        This is a SHAP explanation, it creates local and non-contrastive explanations.

        Returns:
            str: return the explanation method description

        """
        return self.description_text_empty.format(
            self.explanation_name, self.explanation_type, self.explanation_style
        )

    def get_score_text(self) -> str:
        """Generate the text explaining the prediction score of
        the sample

        Returns:
            str: return the score_text for the sample.

        """
        self.number_of_dataset_features = self.X.shape[1]
        return self.score_text_empty.format(
            self.model.__class__.__name__,
            self.number_of_dataset_features,
            self.prediction,
        )

    def get_model_text(self) -> str:
        """
        WIP
        Generate text the explains the used machine learning model

        Returns:
            str: return the description of the machine learning model

        """
        return str(self.model)

    def get_plot_name(self, sample_name: str = None) -> str:
        """
        Get the name of the plot

        Args:
            sample_name (str, optional): [description]. Defaults to None.

        Returns:
            str: return the name of the plot

        """
        prefix = f"{self.explanation_name}_features_{self.number_of_features}"
        if sample_name:
            plot_name = f"{prefix}_sample_{sample_name}.png"
        else:
            plot_name = f"{prefix}.png"
        return plot_name

    def get_sample_name(self, sample_index: int, sample_name: str = None) -> str:
        """
        Determine the name of the sample, if no sample_name provide, use the sample_index

        Args:
            sample_index (int): index of the sample
            sample_name (str, optional): name of the sample. Defaults to None.

        Returns:
            str: name of the sample

        """
        if not sample_name:
            sample_name = str(sample_index)
        return sample_name

    def save(self, sample_index: int, sample_name: str = None) -> None:
        """
        Save the explanations to a csv file, save the plots

        Args:
            sample_index (int): [description]
            sample_name (str, optional): name of the sample. Defaults to None.

        """
        sample_name = self.get_sample_name(sample_index, sample_name)
        self.save_csv(sample_name)
        self.fig.savefig(
            os.path.join(self.path_plot, self.plot_name),
            bbox_inches="tight",
        )

    def save_csv(self, sample: int) -> None:
        """
        Save the explanation to a csv. The columns contain the method_text,
        the natural_language_text, the name of the plot and the predicted
        value. The index is the Entry ID.

        Args:
            sample (TYPE, optional): DESCRIPTION.

        Returns:
            None.

        """
        assert hasattr(self, "plot_name"), "instance lacks plot_name"

        output = {
            "score_text": self.score_text,
            "method_text": self.method_text,
            "natural_language_text": self.natural_language_text,
            "plot": self.plot_name,
            "number_of_features": self.number_of_features,
            "prediction": self.prediction,
        }

        df = pd.DataFrame(output, index=[sample])

        for column in ["method_text", "natural_language_text", "score_text"]:
            df[column] = df[column].astype(str)
            df[column] = df[column].str.replace("\n", "\\n")

        # check if the file is already there, if not, create it
        if not os.path.isfile(os.path.join(self.path_result, self.file_name)):
            df.to_csv(
                os.path.join(self.path_result, self.file_name),
                sep=";",
                encoding="utf-8-sig",
                index_label=["Entry ID"],
                escapechar="\\",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
        else:
            # append row to the file
            df.to_csv(
                os.path.join(self.path_result, self.file_name),
                sep=";",
                encoding="utf-8-sig",
                index_label=["Entry ID"],
                mode="a",
                header=False,
                escapechar="\\",
                quotechar='"',
                quoting=csv.QUOTE_NONNUMERIC,
            )
