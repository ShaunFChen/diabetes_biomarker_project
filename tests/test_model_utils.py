import pytest

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from src.visualization_utils import plot_feature_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

optuna.logging.set_verbosity(optuna.logging.ERROR)

from xgboost import XGBClassifier

from src.model_utils import *


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_prepare_data_with_exclude_columns():
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "outcome": [0, 1, 0],
        "exclude1": ["a", "b", "c"],
    }
    df = pd.DataFrame(data)
    X, y = prepare_data(df, "outcome", exclude_columns=["exclude1"])

    assert "exclude1" not in X.columns
    assert "feature1" in X.columns
    assert "feature2" in X.columns
    assert len(X) == len(y) == 3


def test_prepare_data_without_exclude_columns():
    data = {"feature1": [1, 2, 3], "outcome": [0, 1, 0]}
    df = pd.DataFrame(data)
    X, y = prepare_data(df, "outcome")

    assert "feature1" in X.columns
    assert len(X.columns) == 1
    assert len(X) == len(y) == 3


def test_prepare_data_with_non_existing_exclude_columns():
    data = {"feature1": [1, 2, 3], "outcome": [0, 1, 0]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
        prepare_data(df, "outcome", exclude_columns=["non_existing_column"])


def test_prepare_data_with_missing_outcome_column():
    data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
        prepare_data(df, "outcome")


def test_get_prediction_probabilities_valid_models():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1, 2, 3]})
    y_train = np.array([0, 1, 0])
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [4, 5]})

    models = {"logreg": LogisticRegression()}
    prediction_probs = get_prediction_probabilities(
        models, X_train, y_train, X_test
    )

    assert "logreg" in prediction_probs
    assert len(prediction_probs["logreg"]) == len(X_test)


def test_get_prediction_probabilities_empty_models():
    X_train = pd.DataFrame({"feature1": [1, 2, 3]})
    y_train = np.array([0, 1, 0])
    X_test = pd.DataFrame({"feature1": [4, 5]})

    models = {}
    prediction_probs = get_prediction_probabilities(
        models, X_train, y_train, X_test
    )

    assert prediction_probs == {}


def test_get_prediction_probabilities_invalid_model():
    X_train = pd.DataFrame({"feature1": [1, 2, 3]})
    y_train = np.array([0, 1, 0])
    X_test = pd.DataFrame({"feature1": [4, 5]})

    models = {"invalid_model": "not_a_model"}
    with pytest.raises(TypeError):
        get_prediction_probabilities(models, X_train, y_train, X_test)


def test_exec_basic_logreg_valid_data():
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [1, 2, 3]})
    y_train = np.array([0, 1, 0])

    pipeline = exec_basic_logreg(X_train, y_train)

    assert pipeline is not None
    assert "logreg" in pipeline.named_steps


def test_exec_basic_logreg_non_numeric_data():
    X_train = pd.DataFrame(
        {"feature1": ["a", "b", "c"], "feature2": [1, 2, 3]}
    )
    y_train = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        exec_basic_logreg(X_train, y_train)


def test_exec_basic_logreg_missing_values():
    X_train = pd.DataFrame({"feature1": [1, np.nan, 3], "feature2": [1, 2, 3]})
    y_train = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        exec_basic_logreg(X_train, y_train)


def test_evaluate_model_performance_valid():
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [4, 5]})
    y_test = np.array([0, 1])

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
    )
    pipeline.fit(X_test, y_test)

    y_pred_prob = evaluate_model_performance(pipeline, X_test, y_test)

    assert len(y_pred_prob) == len(y_test)


def test_evaluate_model_performance_mismatched_features():
    X_test = pd.DataFrame({"feature1": [4, 5]})
    y_test = np.array([0, 1])

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
    )
    pipeline.fit(X_test, y_test)

    X_test_new = pd.DataFrame({"feature2": [4, 5]})

    with pytest.raises(ValueError):
        evaluate_model_performance(pipeline, X_test_new, y_test)


def test_evaluate_model_performance_incorrect_labels():
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [4, 5]})
    y_test = np.array([0, 2])  # Incorrect labels

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
    )
    pipeline.fit(X_test, [0, 1])

    with pytest.raises(ValueError):
        evaluate_model_performance(pipeline, X_test, y_test)
