# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 19:34:15 2021

@author: maurol
"""
import os


def create_folder(path):
    """
    create folder, if it doesn't already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path
