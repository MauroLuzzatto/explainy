
<!-- <img src="https://github.com/MauroLuzzatto/explainy/raw/main/docs/_static/logo.png" width="180" height="180" align="right"/> -->
<p align="center">
<img src="https://github.com/MauroLuzzatto/explainy/raw/main/docs/_static/logo.png" width="200" height="200"/>
</p>
<!-- # explainy - machine learning model explanations for humans -->
<!-- # explainy - black-box model explanations for humans -->

<h1 align="center">explainy - black-box model explanations for humans</h1>

[![pypi version](https://img.shields.io/pypi/v/explainy.svg)](https://pypi.python.org/pypi/explainy)
[![codecov](https://codecov.io/gh/MauroLuzzatto/explainy/branch/main/graph/badge.svg?token=N6EKHMEAQR)](https://codecov.io/gh/MauroLuzzatto/explainy)
[![docs](https://readthedocs.org/projects/explainy/badge/?version=latest)](https://explainy.readthedocs.io/en/latest/?version=latest)
[![Supported versions](https://img.shields.io/pypi/pyversions/explainy.svg)](https://pypi.org/project/explainy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Downloads](https://pepy.tech/badge/explainy)](https://pepy.tech/project/explainy)

<!-- [![travis](https://app.travis-ci.com/MauroLuzzatto/explainy.svg?branch=main)](https://app.travis-ci.com/github/MauroLuzzatto/explainy?branch=master) -->


**explainy** is a library for generating machine learning models explanations in Python. It uses methods from **Machine Learning Explainability** and provides a standardized API to create feature importance explanations for samples. 

The API is inspired by `scikit-learn` and has three core methods `explain()`, `plot()` and, `importance()`. The explanations are generated in the form of texts and plots.

**explainy** comes with four different algorithms to create either *global* or *local* and *contrastive* or *non-contrastive* model explanations.


| Method				|Type | Explanations | Classification | Regression | 
| --- 				| --- | :---: | :---: | :---: | 
|[Permutation Feature Importance](https://explainy.readthedocs.io/en/latest/explainy.explanations.html#module-explainy.explanation.permutation_explanation)	| non-contrastive | global |  :star: | :star:|
|[Shap Values](https://explainy.readthedocs.io/en/latest/explainy.explanations.html?highlight=shap#module-explainy.explanations.shap_explanation)		| non-contrastive | local |   	:star: | :star:|
|[Surrogate Model](https://explainy.readthedocs.io/en/latest/explainy.explanations.html#module-explainy.explanation.surrogate_model_explanation)|contrastive | global | :star: | :star: | 
|[Counterfactual Example](https://explainy.readthedocs.io/en/latest/explainy.explanations.html#module-explainy.explanation.counterfactual_explanation)| contrastive | local |:star:| :star:|


Description:
- **global**: explanation of system functionality (all samples have the same explanation)
- **local**: explanation of decision rationale (each sample has its own explanation)
- **contrastive**: tracing of decision path (differences to other outcomes are described)
- **non-contrastive**: parameter weighting (the feature importance is reported)


## Documentation
https://explainy.readthedocs.io


## Install explainy



```
pip install explainy
```

---

Further, install `graphviz` (version 9.0.0 or later) for plotting tree surrogate models:

#### Windows
```
choco install graphviz
```

#### Mac
```
brew install graphviz
```
#### Linux: Ubuntu packages
```
sudo apt install graphviz
```

Further details on how to install `graphviz` can be found in the official [graphviz docs](https://graphviz.org/download/).

Also, make sure that the folder with the `dot` executable is added to your systems `PATH`. You can find further details [here](https://github.com/xflr6/graphviz?tab=readme-ov-file#installation).

## Usage

📚 A comprehensive example of the `explainy` API can be found in this ![Jupyter Notebook](https://github.com/MauroLuzzatto/explainy/blob/main/examples/01-explainy-intro.ipynb) 
 
📖 Or in the [example section](https://explainy.readthedocs.io/en/latest/examples/01-explainy-intro.html) of the documentation


Initialize and train a `scikit-learn` model:
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

model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)
```

Initialize the `PermutationExplanation` (or any other explanation) object and pass in the trained model and the to be explained dataset. 

Addtionally, define the number of features used in the explanation. This allows you to configure the verbosity of your exaplanation.

 Set the index of the sample that should be explained.

```python
from explainy.explanations import PermutationExplanation

number_of_features = 4

explainer = PermutationExplanation(
    X_test, y_test, model, number_of_features
)
```
Call the `explain()` method and print the explanation for the sample (in case of a local explanation every sample has a different explanation).

```python
explanation = explainer.explain(sample_index=1)
print(explanation)
```
> The RandomForestRegressor used 10 features to produce the predictions. The prediction of this sample was 251.8.

> The feature importance was calculated using the Permutation Feature Importance method.

> The four features which were most important for the predictions were (from highest to lowest): 'bmi' (0.15), 's5' (0.12), 'bp' (0.03), and 'age' (0.02).

Use the `plot()` method to create a feature importance plot of that sample.

```python
explainer.plot()
```
![Permutation Feature Importance](https://github.com/MauroLuzzatto/explainy/raw/main/static/permutation_importance.png)

If your prefer, you can also create another type of plot, as for example a boxplot.
```python
explainer.plot(kind='box')
```
![Permutation Feature Importance BoxPlot](https://github.com/MauroLuzzatto/explainy/raw/main/static/permutation_importance_box.png)


Finally, you can also look at the importance values of the features (in form of a `pd.DataFrame`).

```python
feature_importance = explainer.importance()
print(feature_importance)
```

```python
  Feature  Importance
0     bmi        0.15
1      s5        0.12
2      bp        0.03
3     age        0.02
4      s2       -0.00
5     sex       -0.00
6      s3       -0.00
7      s1       -0.01
8      s6       -0.01
9      s4       -0.01
```

<!-- Finally the result can be saved

```python
explainer.save(sample_index)
``` -->

<!-- 
## Model Explanations
-->


## Features
- Algorithms for inspecting black-box machine learning models 
- Support for the machine learning frameworks `scikit-learn` and `xgboost`
- **explainy** offers a standardized API with three core methods `explain()`, `plot()`, `importance()`

## Other Machine Learning Explainability libraries to watch
- [shap](https://github.com/slundberg/shap): A game theoretic approach to explain the output of any machine learning model
- [eli5](https://github.com/TeamHG-Memex/eli5): A library for debugging/inspecting machine learning classifiers and explaining their predictions 
- [alibi](https://github.com/SeldonIO/alibi): Algorithms for explaining machine learning models 
- [interpret](https://github.com/interpretml/interpret): Fit interpretable models. Explain blackbox machine learning


## Source

Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019. https://christophm.github.io/interpretable-ml-book/

## Author
**Mauro Luzzatto** - [Maurol](https://github.com/MauroLuzzatto)

