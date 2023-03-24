import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


def get_regression_model():
    diabetes = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=diabetes.feature_names)
    y_test = pd.DataFrame(y_test)

    model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
    return model, X_test, y_test


def get_classification_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=0
    )
    X_test = pd.DataFrame(X_test, columns=iris.feature_names)
    y_test = pd.DataFrame(y_test)

    model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    return model, X_test, y_test
