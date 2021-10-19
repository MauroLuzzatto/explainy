# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 21:58:56 2021

@author: maurol
"""

import pytest

from explainy.core.explanation import Explanation
from ..utils import get_classification_model


def test_Explanation():

    score_text = 'a'
    method_text = 'b'
    natural_language_text = 'c'

    explanation = Explanation(
        score_text, method_text, natural_language_text, separator='-'
    )
    assert explanation.explanation == "a-b-c"
    assert explanation.score_text == score_text
    assert explanation.method_text == method_text
    assert explanation.natural_language_text == natural_language_text



if __name__ == "__main__":
    pytest.main()
    
