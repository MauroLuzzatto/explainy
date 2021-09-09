================================================
explainy - model explanations for humans
================================================


.. image:: https://img.shields.io/pypi/v/explainy.svg
        :target: https://pypi.python.org/pypi/explainy

.. image:: https://app.travis-ci.com/MauroLuzzatto/explainy.svg?branch=main
        :target: https://app.travis-ci.com/github/MauroLuzzatto/explainy?branch=master


.. image:: https://readthedocs.org/projects/explainy/badge/?version=latest
        :target: https://explainy.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://img.shields.io/pypi/pyversions/explainy.svg
    :alt: Supported versions
    :target: https://pypi.org/project/explainy


.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
    :alt: Code style: black
    :target: https://github.com/ambv/black




explainy is a library for generating machine learning models explanations in Python. It uses methods from **Machine Learning Explainability** and provides a standardized API to create feature importance explanations for samples. The explanations are generated in the form of plots and text.


* Free software: MIT license

Documentation 
--------------
https://explainy.readthedocs.io


Usage
------

```python
from explainy.explanations import PermutationExplanation

explainer = PermutationExplanation(
	X, y, model, number_of_features=10
)

for sample_index in range(10):
    explanation = explainer.explain(sample_index=sample_index)
    explainer.print_output()
    explainer.plot()
    explainer.save(sample_index)

```

Features
--------

* TODO

