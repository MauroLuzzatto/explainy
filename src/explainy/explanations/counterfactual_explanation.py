"""Counterfactual Explanation
--------------------------
Counterfactual explanations tell us how the values of an instance have to change to 
significantly change its prediction. A counterfactual explanation of a prediction 
describes the smallest change to the feature values that changes the prediction 
to a predefined output. By creating counterfactual instances, we learn about how the 
model makes its predictions and can explain individual predictions [1].

Characteristics
===============
- local
- contrastive

Source
======
[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. 
https://christophm.github.io/interpretable-ml-book/
"""

import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from mlxtend.evaluate import create_counterfactual
from sklearn.base import is_regressor

from explainy.core.explanation import Explanation
from explainy.core.explanation_base import ExplanationBase
from explainy.utils.typing import Config, ModelType
from explainy.utils.utils import join_text_with_comma_and_and, num_to_str

np.seterr(divide="ignore", invalid="ignore")

COLUMN_REFERENCE = "Reference Values"
COLUMN_COUNTERFACTUAL = "Counterfactual Values"
COLUMN_DIFFERENCE = "Prediction Difference"


class CounterfactualExplanation(ExplanationBase):
    """Contrastive, local Explanation"""

    explanation_type: str = "local"
    explanation_style: str = "contrastive"
    explanation_name: str = "counterfactual"

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model: ModelType,
        y_desired: float,
        number_of_features: int = 4,
        config: Optional[Config] = None,
        delta: Optional[float] = None,
    ) -> None:
        super(CounterfactualExplanation, self).__init__(model, config)
        """
        This implementation is a thin wrapper around `smlxtend.evaluate.create_counterfactual
        <http://rasbt.github.io/mlxtend/user_guide/evaluate/create_counterfactual>`

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (sklearn.base.BaseEstimator): trained (sckit-learn) model object
            number_of_features (int): number of features to consider in the explanation
            
            config (dict): configuration dictionary
            y_desired (float, optional): desired target value for the counter factual example. Defaults to max(y).
            delta (float, optional): maximum allowed difference between the desired target value and the predicted value. Defaults to prediction * 0.05.
            random_state (int, optional): random state for the counter factual example. Defaults to 0.
            
        Returns:
            None.
        """
        self.X = X
        self.y = y
        self.y_desired = y_desired
        self.delta = delta
        self.feature_names = self.get_feature_names(self.X)
        self.number_of_features = self.get_number_of_features(number_of_features)
        self.sample_index: int = None

        self.is_regressor = is_regressor(self.model)

        natural_language_text_empty = (
            "The sample would have had the desired prediction of '{}', {}."
        )
        method_text_empty = (
            "The feature importance is shown using a counterfactual example."
        )
        sentence_text_empty = "the '{}' was {}"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )
        self.logger = self.setup_logger(self.explanation_name)

    def _calculate_importance(
        self, sample_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create the counter factual explanation for the given sample.

        Args:
            sample_index (int, optional): sample index. Defaults to 0.

        Returns:
            x_ref (np.ndarray): reference feature values
            x_counter_factual (np.ndarray): counter factual feature values
        """
        x_ref = self.X.values[sample_index, :]

        if not self.delta:
            if self.is_regressor:
                self.delta = self.prediction * 0.05
            else:
                # in case of a classifier, we are trying the find the right class
                self.delta = 0

            self.logger.info(
                f"No delta value set, therefore using the value '{self.delta}'"
            )

        start = -2
        stop = 2
        num = stop - start + 1

        self.logger.info(
            "Start to calculate the counterfactual example. This may take a while..."
        )

        is_value_found = False
        # try different seed values
        for random_seed in range(14):
            # use exponential increase to search for the right lammbda value
            for lammbda in np.logspace(
                start=start,
                stop=stop,
                num=num,
                base=10,
                dtype="float",
            ):
                # catch the warning "Maximum number of function evaluations has been exceeded." warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    x_counter_factual = create_counterfactual(
                        x_reference=x_ref,
                        y_desired=self.y_desired,
                        model=self.model,
                        X_dataset=self.X.values,
                        lammbda=lammbda,
                        random_seed=random_seed,
                    )

                    self.y_counter_factual = self.model.predict(
                        x_counter_factual.reshape(1, -1)
                    )[0]

                local_delta = np.abs(self.y_counter_factual - self.y_desired)

                self.logger.info(
                    f"y_counter_factual: {self.y_counter_factual:.2f}, lambda:"
                    f" {lammbda}, local_delta: {local_delta}, random_seed:"
                    f" {random_seed}"
                )
                self.logger.debug(
                    f" y_desired: {self.y_desired:.2f}, y_pred:"
                    f" {self.prediction:.2f}, label:"
                    f" {self.y.values[sample_index]}, delta:"
                    f" {self.delta}, "
                )

                if self.is_regressor:
                    if local_delta < self.delta:
                        self.logger.debug("found value below delta!")
                        is_value_found = True
                        break
                else:
                    if self.y_counter_factual == self.y_desired:
                        self.logger.debug("found the right class!")
                        is_value_found = True
                        break

            if is_value_found:
                break

        else:
            raise ValueError(
                "No counterfactual value found, try to decrease the 'delta'"
                " value or adjust the desired prediction 'y_desired'"
            )

        self.logger.debug(f"Features of the sample: {x_ref}")
        self.logger.debug(f"Features of the countefactual: {x_counter_factual}")

        return x_ref, x_counter_factual

    def get_prediction_from_new_feature_value(
        self,
        feature_index: int,
        x_ref: np.ndarray,
        x_counter_factual: np.ndarray,
    ) -> float:
        """Replace the value of the feauture at the postition of thw feature_index and predict
        a new value for this new set of features

        Args:
            feature_index (int): The index of the feature to replace with the counterfactual value.
            x_ref (np.ndarray): reference features.
            x_counter_factual (np.ndarray): counter factual features.

        Returns:
            prediction (float): predicted value with the updated features values.

        """
        x_created = x_ref.reshape(1, -1).copy()
        old_value = x_created[0, feature_index]
        new_value = x_counter_factual.reshape(1, -1)[0, feature_index]
        self.logger.debug(f"old_value: {old_value:.4f}, new_value: {new_value:.4f}")

        # assign new value
        x_created[0, feature_index] = x_counter_factual.reshape(1, -1)[0, feature_index]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred_new = self.model.predict(x_created)[0]
        return pred_new

    def get_feature_importance(
        self, x_ref: np.ndarray, x_counter_factual: np.ndarray
    ) -> list:
        """Calculate the importance of each feature. Take the reference
        features and replace every feature with the new counter_factual value.
        Calculat the absulte difference that this feature adds to the prediction.
        A larger absolute value, means a larger contribution and therefore a more
        important feature.

        Args:
            x_ref (np.ndarray): reference features.
            x_counter_factual (np.ndarray): counter factual features.

        Returns:
            list: list of the feature sorted by importance
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred_ref = self.model.predict(x_ref.reshape(1, -1))[0]

        self.differences = []
        for feature_index in range(x_ref.shape[0]):
            pred_new = self.get_prediction_from_new_feature_value(
                feature_index, x_ref, x_counter_factual
            )
            difference = pred_new - pred_ref
            self.differences.append(difference)
            self.logger.debug(
                f"name: {self.feature_names[feature_index]}, difference:"
                f" {self.differences[feature_index]:.2f}"
            )
        # get the sorted feature_names
        self.feature_sort = np.array(self.feature_names)[
            np.array(self.differences).argsort()[::-1]
        ].tolist()
        return self.feature_sort

    def get_feature_values(
        self,
        x_ref: np.ndarray,
        x_counter_factual: np.ndarray,
        decimal: int = 2,
        debug: bool = False,
    ):
        """Arrange the reference and the counter factual features in a dataframe

        Args:
            x_ref (np.array): features of the sample
            x_counter_factual (np.array): features of the counter factual sample to achive y_desired
            decimal (int): decimal number to round the values to in the plot
            debug (bool): if True, plot the dataframe

        Returns:
            None.
        """
        index = [
            COLUMN_REFERENCE,
            COLUMN_COUNTERFACTUAL,
            COLUMN_DIFFERENCE,
        ]
        self.df = (
            pd.DataFrame(
                [x_ref, x_counter_factual, self.differences],
                index=index,
                columns=self.feature_names,
            )
            .round(decimal)
            .T
        )
        # reorder dataframe according the the feature importance
        self.df = self.df.loc[self.feature_sort, :]
        try:
            self.df[COLUMN_DIFFERENCE][self.df[COLUMN_DIFFERENCE] != 0]
        except IndexError as e:
            print(e)

        if debug:
            self.df.plot(kind="barh", figsize=(3, 5))

    def importance(self) -> pd.DataFrame:
        """Return the feature importance

        Returns:
            pd.DataFrame: dataframe with the feature importance
        """
        return self.df.round(2)

    def plot(self, sample_index: int, kind: str = "table") -> None:
        """Create the plot of the counterfactual table

        Args:
            kind (str, optional): kind of plot. Defaults to 'table'.

        Raises:
            Exception: raise Exception if the "kind" of plot is not supported

        """
        if sample_index != self.sample_index:
            raise ValueError(
                "sample_index is not the same as the index used to calculate the"
                " counterfactual explanation, re-run .explain(sample_index) to plot the"
                " correct sample"
            )
        if kind == "table":
            self.fig = self._plot_table()
        else:
            raise Exception(f'Value of kind "{kind}" is not supported!')

    def _plot_table(self) -> plt.figure:
        """Plot the table comparing the refence and the counterfactual values

        Returns:
            plt.figure: figure object

        """
        colLabels = ["Sample", "Counterfactual Sample"]
        columns = [COLUMN_REFERENCE, COLUMN_COUNTERFACTUAL]

        array_subset = self.df[columns].values[: self.number_of_features]
        rowLabels = list(self.df.index)[: self.number_of_features]

        # if show_rating:
        score_row = np.array(
            [f"{self.prediction:.1f}", f"{self.y_counter_factual:.1f}"]
        ).reshape(1, -1)

        array_subset = np.append(array_subset, score_row, axis=0)
        rowLabels = rowLabels + ["Prediction"]

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")

        table = ax.table(
            cellText=array_subset,
            colLabels=colLabels,
            rowLabels=rowLabels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.25, 2)

        # if show_rating:
        # make the last row bold
        for (row, _), cell in table.get_celld().items():
            if row == array_subset.shape[0]:
                cell.set_text_props(fontproperties=FontProperties(weight="bold"))

        plt.axis("off")
        plt.grid("off")
        plt.tight_layout()
        plt.show()
        return fig

    def get_method_text(self) -> str:
        """Define the method introduction text of the explanation type.

        Returns:
            str: method text explanation
        """
        return self.method_text_empty.format(
            num_to_str[self.number_of_features], self.y_counter_factual
        )

    def get_natural_language_text(self) -> str:
        """Define the natural language output using the feature names
        and its values for this explanation type

        Returns:
            str: natural language explanation

        """
        feature_values = self.df[COLUMN_COUNTERFACTUAL].tolist()[
            : self.number_of_features
        ]
        feature_names = list(self.df.index)[: self.number_of_features]

        sentences = []
        for feature_name, feature_value in zip(feature_names, feature_values):
            sentence_filled = self.sentence_text_empty.format(
                feature_name, f"'{feature_value}'"
            )
            sentences.append(sentence_filled)

        sentences = "if " + join_text_with_comma_and_and(sentences)
        natural_language_text = self.natural_language_text_empty.format(
            self.y_counter_factual, sentences
        )
        return natural_language_text

    def _setup(self, sample_index: int, sample_name: str) -> None:
        """Helper function to setup the counterfactual explanation

        Args:
            sample_index (int): index of sample in scope
            sample_name (str): name of the sample in scope

        Returns:
            None
        """
        x_ref, x_counter_factual = self._calculate_importance(sample_index)
        self.get_feature_importance(x_ref, x_counter_factual)
        self.get_feature_values(x_ref, x_counter_factual)
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()
        self.plot_name = self.get_plot_name(sample_name)

    def explain(
        self,
        sample_index: int,
        sample_name: Optional[str] = None,
        separator: str = "\n",
    ) -> Explanation:
        """Main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample_index (int): number of the sample to create the explanation for
            sample_name (str, optional): name of the sample. Defaults to None.
            separator (str, optional): separator for the natural language text. Defaults to "\n".

        Returns:
            Explanation: Explanation object containg the explainations
        """
        self.sample_index = sample_index
        sample_name = self.get_sample_name(sample_index, sample_name)
        self.prediction = self.get_prediction(sample_index)
        self.score_text = self.get_score_text()

        self._setup(sample_index, sample_name)
        self.explanation = Explanation(
            self.score_text,
            self.method_text,
            self.natural_language_text,
            separator=separator,
        )
        return self.explanation
