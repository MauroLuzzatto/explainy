# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:49:43 2021

@author: maurol
"""
import os

from src.explanation.CategoryMapper import CategoryMapper


class ExplanationMixin:
    def map_category(self, feature_name, feature_value):
        """


        Args:
            feature_name (TYPE): DESCRIPTION.
            feature_value (TYPE): DESCRIPTION.

        Returns:
            feature_value (TYPE): DESCRIPTION.

        """
        # TODO: fix path
        path_json = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\model_explanation_study\dataset\training\mapping"
        if f"{feature_name}.json" in os.listdir(path_json):
            mapper = CategoryMapper(path_json, feature_name)
            # print(mapper[int(feature_value)])
            feature_value = mapper[int(feature_value)]
        return feature_value

    @staticmethod
    def join_text_with_comma_and_and(values: list) -> str:
        """
        Merge values for text output with commas and only the last value
        with an "and""

        Args:
            values (list): list of values to be merged.

        Returns:
            str: new text.

        """

        if len(values) > 2:
            last_value = values[-1]
            values = ", ".join(values[:-1])
            text = values + ", and " + last_value

        else:
            text = ", and ".join(values)
        return text

    def get_number_to_string_dict(self) -> None:
        """
        map number of features to string values
        """

        number_text = (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        )
        self.num_to_str = {}
        for text, number in zip(number_text, range(1, 21)):
            self.num_to_str[number] = text
