"""
Counterfactual Explanation
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
import os
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.font_manager import FontProperties
from mlxtend.evaluate import create_counterfactual

from explainy.core.explanation_base import ExplanationBase

# np.seterr(divide="ignore", invalid="ignore")
RANDOM_SEED = 0


class CounterfactualExplanation(ExplanationBase):
    """
    Contrastive, local Explanation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.array,
        model: sklearn.base.BaseEstimator,
        number_of_features: int = 4,
        config: Dict = None,
        y_desired: float = None,
    ) -> None:
        super(CounterfactualExplanation, self).__init__(config)
        """
        This implementation is a thin wrapper around `smlxtend.evaluate.create_counterfactual
        <http://rasbt.github.io/mlxtend/user_guide/evaluate/create_counterfactual>`

        Args:
            X (df): (Test) samples and features to calculate the importance for (sample, features)
            y (np.array): (Test) target values of the samples (samples, 1)
            model (object): trained (sckit-learn) model object
            number_of_features (int): 
            y_desired (float, optional): desired target value for the counter factual example. 
                Defaults to max(y).
        
        Returns:
            None.

        """
        self.X = X
        self.y = y

        if y_desired is None:
            y_desired = y.values.max()

        self.y_desired = y_desired
        self.model = model
        self.feature_names = list(X)
        self.number_of_features = number_of_features

        natural_language_text_empty = (
            "The sample would have had the desired prediction, {}."
        )

        method_text_empty = (
            "The feature importance is shown using a counterfactual example."
        )

        sentence_text_empty = "the '{}' was {}"

        self.define_explanation_placeholder(
            natural_language_text_empty, method_text_empty, sentence_text_empty
        )

        self.explanation_name = "counterfactual"
        self.logger = self.setup_logger(self.explanation_name)

        self.delta = self.config.get("delta", 1.0)

    def _calculate_importance(self, sample=0, delta=1.0):
        """
        Create the counter factual explanation for the given sample.

        Args:
            sample (int, optional): DESCRIPTION. Defaults to 0.
            lammbda (float, optional): hyperparameter (0,1). Defaults to 1.0.

        Returns:
            x_ref (TYPE): DESCRIPTION.
            x_counter_factual (TYPE): DESCRIPTION.

        """

        x_ref = self.X.values[sample, :]

        count = 0
        for lammbda in np.arange(0, 10000, 0.1):

            x_counter_factual = create_counterfactual(
                x_reference=x_ref,
                y_desired=self.y_desired,
                model=self.model,
                X_dataset=self.X.values,
                y_desired_proba=None,
                lammbda=lammbda,
                random_seed=RANDOM_SEED,
            )

            self.y_counter_factual = self.model.predict(
                x_counter_factual.reshape(1, -1)
            )[0]

            self.log_counterfactual(lammbda)

            if np.abs(self.y_counter_factual - self.y_desired) < delta:
                break

            if count > 100:
                raise

            count += 1

        self.logger.info("\nFinal Lambda:")
        self.log_counterfactual(lammbda)
        self.log_output(sample, x_ref, x_counter_factual)
        return x_ref, x_counter_factual

    def log_counterfactual(self, lammbda):
        """


        Args:
            lammbda (TYPE): DESCRIPTION.

        Returns:
            None.

        """

        self.logger.info(f"lambda: {lammbda}")
        self.logger.info(
            f"diff: {np.abs(self.y_counter_factual - self.y_desired)}"
        )
        self.logger.info(
            f"y_counterfactual: {self.y_counter_factual:.2f}, desired:"
            f" {self.y_desired:.2f}, y_pred {self.prediction:.2f}"
        )
        self.logger.info("---" * 15)

    def log_output(self, sample, x_ref, x_counter_factual):
        """


        Args:
            sample (TYPE): DESCRIPTION.
            x_ref (TYPE): DESCRIPTION.
            x_counter_factual (TYPE): DESCRIPTION.

        Returns:
            None.

        """

        self.logger.info("True label: {}".format(self.y.values[sample]))
        self.logger.info(
            "Predicted label: {}".format(
                self.model.predict(x_ref.reshape(1, -1))[0]
            )
        )
        self.logger.info(f"Desired label: {self.y_desired}")
        self.logger.info("Features of the sample: {}".format(x_ref))
        self.logger.info(
            "Features of the countefactual: {}".format(x_counter_factual)
        )
        self.logger.info("Predictions for counterfactual:")
        self.logger.info("Desired label: {}".format(self.y_desired))
        self.logger.info(
            "Predicted label: {}".format(
                self.model.predict(x_counter_factual.reshape(1, -1))[0]
            )
        )

    def get_prediction_from_new_value(self, ii, x_ref, x_counter_factual):
        """
        replace the value of the feauture at postion ii and predict
        a new value for this new set of features

        Args:
            ii (TYPE): DESCRIPTION.
            x_ref (TYPE): DESCRIPTION.
            x_counter_factual (TYPE): DESCRIPTION.

        Returns:
            difference (TYPE): DESCRIPTION.

        """
        x_created = x_ref.reshape(1, -1).copy()
        old_value = x_created[0, ii]
        new_value = x_counter_factual.reshape(1, -1)[0, ii]
        # assign new value
        x_created[0, ii] = x_counter_factual.reshape(1, -1)[0, ii]
        self.logger.debug(f"old_value: {old_value} -- new_value: {new_value}")
        pred_new = self.model.predict(x_created)[0]
        return pred_new

    def get_feature_importance(self, x_ref, x_counter_factual):
        """
        Calculate the importance of each feature. Take the reference
        features and replace every feature with the new counter_factual value.
        Calculat the absulte difference that this feature adds to the prediction.
        A larger absolute value, means a larger contribution and therefore a more
        important feature.

        Args:
            x_ref (TYPE): DESCRIPTION.
            x_counter_factual (TYPE): DESCRIPTION.

        Returns:
            None.
        """
        pred_ref = self.model.predict(x_ref.reshape(1, -1))[0]

        self.differences = []
        for ii in range(x_ref.shape[0]):

            pred_new = self.get_prediction_from_new_value(
                ii, x_ref, x_counter_factual
            )

            difference = pred_new - pred_ref

            # print(difference, pred_new, pred_ref)
            self.differences.append(difference)
            self.logger.debug(
                "name: {} -- difference: {}".format(
                    self.feature_names[ii], self.differences[ii]
                )
            )
        # get the sorted feature_names
        self.feature_sort = np.array(self.feature_names)[
            np.array(self.differences).argsort()[::-1]
        ].tolist()

    def get_feature_values(self, x_ref, x_counter_factual, decimal=2):
        """
        Arrange the reference and the counter factual features in a dataframe

        Args:
            x_ref (np.array): features of the sample
            x_counter_factual (np.array): features of the counter factual sample to achive y_desired
            decimal (int): decimal number to round the values to in the plot

        Returns:
            None.

        """
        self.df = (
            pd.DataFrame(
                [x_ref, x_counter_factual, self.differences],
                index=[
                    "Reference Values",
                    "Counter Factual Values",
                    "difference of the new feature in the prediction",
                ],
                columns=self.feature_names,
            )
            .round(decimal)
            .T
        )
        # reorder dataframe according the the feature importance
        self.df = self.df.loc[self.feature_sort, :]

        try:
            self.df["difference of the new feature in the prediction"][
                self.df["difference of the new feature in the prediction"] != 0
            ].plot(kind="barh", figsize=(3, 5))
        except IndexError as e:
            print(e)

    def format_features_for_plot(self):
        """
        - map categorical variables
        - replace one-hot-encoded value with True, False strings

        Returns:
            None.

        """
        for feature_name in list(self.df.index)[: self.number_of_features]:
            for col_name in ["Reference Values", "Counter Factual Values"]:

                feature_value = self.df.loc[feature_name, col_name]
                self.df.loc[feature_name, col_name] = self.map_category(
                    feature_name, feature_value
                )

                # replace one-hot-encoded value with True, False strings
                if " - " in feature_name:
                    self.logger.debug(
                        f"{feature_name}, {col_name},"
                        f" {self.df.loc[feature_name, col_name]}"
                    )
                    if self.df.loc[feature_name, col_name] == 1.0:
                        string = "Yes"  # "True"
                    else:
                        string = "No"  # "False"

                    self.df.loc[feature_name, col_name] = string

    def plot(self, show_rating=False):
        """
        Plot the table comparing the refence and the counter factual values

        Returns:
            None.
        """

        colLabels = ["Sample", "Counterfactual Values"]
        columns = ["Reference Values", "Counter Factual Values"]

        self.format_features_for_plot()
        array_subset = self.df[columns].values[: self.number_of_features]
        rowLabels = list(self.df.index)[: self.number_of_features]

        if show_rating:
            score_row = np.array(
                [f"{self.prediction:.1f}", f"{self.y_counter_factual:.1f}"]
            ).reshape(1, -1)
            array_subset = np.append(array_subset, score_row, axis=0)
            rowLabels = rowLabels + ["Application rating"]

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

        if show_rating:

            # make the last row bold
            for (row, col), cell in table.get_celld().items():
                if row == array_subset.shape[0]:
                    cell.set_text_props(
                        fontproperties=FontProperties(weight="bold")
                    )

        plt.axis("off")
        plt.grid("off")
        # draw canvas once
        plt.gcf().canvas.draw()
        # get bounding box of table
        points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
        # add 10 pixel spacing
        points[0, :] -= 10
        points[1, :] += 10

        # get new bounding box in inches
        self.nbbox = matplotlib.transforms.Bbox.from_extents(
            points / plt.gcf().dpi
        )
        plt.show(block=True)
        return fig

    def save(self, sample_index, sample_name):

        self.save_csv(sample_name)

        self.fig.savefig(
            os.path.join(self.path_plot, self.plot_name),
            bbox_inches=self.nbbox,
        )

    def get_method_text(self):
        """
        Define the method introduction text of the explanation type.

        Returns:
            None.
        """
        return self.method_text_empty.format(
            self.num_to_str[self.number_of_features], self.y_counter_factual
        )

    def get_natural_language_text(self):
        """
        Define the natural language output using the feature names
        and its values for this explanation type

        Returns:
            None.
        """
        feature_values = self.df["Counter Factual Values"].tolist()[
            : self.number_of_features
        ]
        feature_names = list(self.df.index)[: self.number_of_features]

        sentences = []
        for feature_name, feature_value in zip(feature_names, feature_values):
            feature_value = self.map_category(feature_name, feature_value)

            # handle one-hot encoding case
            if " - " in feature_name:
                sentence_filled = self.create_one_hot_sentence(
                    feature_name, feature_value, self.sentence_text_empty
                )
                mode = "one-hot feature"
            else:
                sentence_filled = self.sentence_text_empty.format(
                    feature_name, f"'{feature_value}'"
                )
                mode = "standard feature"

            self.logger.debug(f"{mode}: {sentence_filled}")
            sentences.append(sentence_filled)

        sentences = "if " + self.join_text_with_comma_and_and(sentences)
        return self.natural_language_text_empty.format(sentences)

    @staticmethod
    def create_one_hot_sentence(feature_name, feature_value, sentence):
        """
        Create sentence from one-hot-encoded feature value, split the
        column name into feature and value and create sentence
        based on if the value was 1 = True, or 0 = False

        Args:
            feature_name (TYPE): DESCRIPTION.
            feature_value (TYPE): DESCRIPTION.
            sentence (TYPE): DESCRIPTION.

        Returns:
            sentence_filled (TYPE): DESCRIPTION.

        """

        column, value = feature_name.split(" - ")

        if int(feature_value) == 1:
            sentence_filled = sentence.format(column, f"'{value}'")
        else:
            sentence_filled = sentence.format(column, f"not '{value}'")

        return sentence_filled

    def _setup(self, sample_index, sample_name):

        self.get_prediction(sample_index)

        x_ref, x_counter_factual = self._calculate_importance(
            sample_index, delta=self.delta
        )
        self.get_feature_importance(x_ref, x_counter_factual)
        self.get_feature_values(x_ref, x_counter_factual)
        self.natural_language_text = self.get_natural_language_text()
        self.method_text = self.get_method_text()

        # todo: move to save?
        self.plot_name = self.get_plot_name(sample_name)

    def explain(self, sample_index, sample_name=None, separator="\n"):
        """
        main function to create the explanation of the given sample. The
        method_text, natural_language_text and the plots are create per sample.

        Args:
            sample (int): number of the sample to create the explanation for

        Returns:
            None.
        """
        if not sample_name:
            sample_name = sample_index

        self._setup(sample_index, sample_name)

        self.score_text = self.get_score_text()
        self.explanation = self.get_explanation(separator)
        return self.explanation
