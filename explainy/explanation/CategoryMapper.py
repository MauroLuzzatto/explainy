# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:39:26 2021

@author: maurol
"""
import json
import os


class CategoryMapper(object):
    """
    map the integer of the categorical variables back to the orginal string
    using the saved json file
    """

    def __init__(self, path, feature):
        self.path_json = os.path.join(path, f"{feature}.json")
        self.get_map_index_dict()

    def load_json(self):
        """
        load the json file by path

        Returns:
            mapper (TYPE): DESCRIPTION.

        """
        with open(self.path_json, "r", encoding="utf-8") as json_file:
            mapper = json.load(json_file)
        return mapper

    @staticmethod
    def key_to_int(mapper):
        """
        convert the key of the json file to an integer

        Args:
            mapper (TYPE): DESCRIPTION.

        Returns:
            dict: DESCRIPTION.

        """
        return {int(k): v for k, v in mapper.items()}

    def get_map_index_dict(self):
        """
        create the dictionary containing the index as key and the
        string as value

        Returns:
            None.

        """
        mapper = self.load_json()
        self.map_index_dict = self.key_to_int(mapper)

    def __getitem__(self, index):
        assert type(index) == int, "index is not an integer"
        return self.map_index_dict[index]

    def __call__(self, index):
        assert type(index) == int, "index is not an integer"
        return self.map_index_dict[index]


if __name__ == "__main__":

    path = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\model_explanation_study\dataset\training"
    feature = "State of residence"
    mapper = CategoryMapper(path, feature)
    print(mapper[3])

    feature = "Favorite subjects in school"
    mapper = CategoryMapper(path, feature)
    print(mapper[3])
