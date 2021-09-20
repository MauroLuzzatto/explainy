
<img src="https://github.com/MauroLuzzatto/explainy/raw/main/docs/_static/logo.png" width="170" height="170" align="right" /></a>
# explainy - black-box model explanations for humans


[![pypi version](https://img.shields.io/pypi/v/explainy.svg)](https://pypi.python.org/pypi/explainy)
[![travis](https://app.travis-ci.com/MauroLuzzatto/explainy.svg?branch=main)](https://app.travis-ci.com/github/MauroLuzzatto/explainy?branch=master)
[![docs](https://readthedocs.org/projects/explainy/badge/?version=latest)](https://explainy.readthedocs.io/en/latest/?version=latest)
[![Supported versions](https://img.shields.io/pypi/pyversions/explainy.svg)](https://pypi.org/project/explainy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![Downloads](https://pepy.tech/badge/explainy)](https://pepy.tech/project/explainy)



explainy is a library for generating explanations for machine learning models in Python. It uses methods from **Machine Learning Explainability** and provides a standardized API to create feature importance explanations for samples. The explanations are generated in the form of plots and text.

explainy comes with four different algorithms to create either global or local and contrastive or non-contrastive machine learning model explanations.


## Documentation
https://explainy.readthedocs.io


## Install explainy

```
pip install explainy
```

## Usage

Create and train a `sklearn` model:
```python
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=0
)
X_test = pd.DataFrame(X_test, columns=diabetes.feature_names)
y_test = pd.DataFrame(y_test)

model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
```

Pass the trained model and the - to be explained - samples into a `PermuationExplaination` (or any other explanation) object. Defined the number of features used in the explanation as well as the index of the sample that should be explained.

```python
from explainy.explanation.permutation_explanation import PermutationExplanation

number_of_features = 4
sample_index = 1

explainer = PermutationExplanation(
    X_test, y_test, model, number_of_features
)

explanation = explainer.explain(
    sample_index, separator='\n'
)
```
Print out the explanation of the sample in scope:

```python
print(explanation)
```
> The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8.

> The feature importance was calculated using the Permutation Feature Importance method.

> The four features which were most important for the predictions were (from highest to lowest): 'bmi' (0.15), 's5' (0.12), 'bp' (0.03), and 'age' (0.02).

Plot the feature importance of that sample:

```python
explainer.plot()
```
![Permutation Feature Importance](static/permutation_importance.png)
```python
explainer.save(sample_index)
```


## Model Explanations

| Method				|Type | Explanations | Classification | Regression | 
| --- 				| --- | --- | :---: | :---: | 
|[Permutation Feature Importance](https://explainy.readthedocs.io/en/latest/explainy.explanation.html#module-explainy.explanation.permutation_explanation)	| Non-contrastive | global |  :star: | :star:|
| Shapley Values		| Non-contrastive | local |   :star: | :star:|
|Global Surrogate Model|Contrastive | global | :star: | :star:|
|Counterfactual Example| Contrastive | local |:star: | :star:|


Description
- global: 
- local:
- contrastive:	
- non-contrastive:	



## Features

TODO

<!-- 

## Explanations

### Permutation Feature Importance

Permutation feature importance measures the increase in the prediction error of the model after we permuted the feature's values, which breaks the relationship between the feature and the true outcome [1].

#### Characteristics
- global
- non-contrastive

![Permutation Feature Importance](docs/_static/permutation.png)

### Shapley Values
A prediction can be explained by assuming that each feature value of  the instance is a "player" in a game where the prediction is the payout.  Shapley values (a method from coalitional game theory) tells us how  to fairly distribute the "payout" among the features. The Shapley value is the average marginal contribution of a feature value across all possible coalitions [1].

#### Characteristics
- local
- non-contrastive

![Shapley Values](docs/_static/shapley.png)


### Counterfactual explanations
Counterfactual explanations tell us how the values of an instance have to change to significantly change its prediction. A counterfactual explanation of a prediction describes the smallest change to the feature values that changes the prediction to a predefined output. By creating counterfactual instances, we learn about how the model makes its predictions and can explain individual predictions [1].

#### Characteristics
- local
- contrastive

![Counterfactual Example](docs/_static/counterfactual.png)


### Global Surrogate Model (Decision Tree)
A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model. We can draw conclusions about the black box model by interpreting the surrogate model [1].

#### Characteristics
- global
- contrastive

![Global Surrogate Model](docs/_static/surrogate.png)
 -->

## Source

[1] Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. https://christophm.github.io/interpretable-ml-book/

## Authors
* **Mauro Luzzatto** - [Maurol](https://github.com/MauroLuzzatto)

