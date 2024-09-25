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
from lifelines import CoxPHFitter
from sklearn.datasets import make_classification

import warnings
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


def test_optimize_linear_model():
    # Generate synthetic data
    np.random.seed(0)
    n_samples = 100
    X_train = pd.DataFrame(
        {
            "feature1": np.random.normal(size=n_samples),
            "feature2": np.random.normal(size=n_samples),
            "feature3": np.random.normal(size=n_samples),
        }
    )
    y_train = np.random.binomial(1, 0.5, size=n_samples)

    # Define parameter grid
    param_grid = {
        "estimator__C": [0.1, 1, 10],
        "estimator__penalty": ["l1", "l2"],
        "estimator__solver": ["liblinear"],
    }

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define estimator
    estimator = LogisticRegression(max_iter=1000)

    # Call the function
    best_pipeline, selected_features, best_hyperparams = optimize_linear_model(
        X_train, y_train, param_grid, cv, estimator, scoring="f1"
    )

    # Assertions
    assert best_pipeline is not None, "The best_pipeline should not be None."
    assert isinstance(
        selected_features, list
    ), "selected_features should be a list."
    assert isinstance(
        best_hyperparams, dict
    ), "best_hyperparams should be a dictionary."
    assert (
        len(selected_features) > 0
    ), "At least one feature should be selected."
    print("Test passed for optimize_linear_model.")


def test_optuna_xgb_obj():
    # Generate synthetic data
    np.random.seed(0)
    X_train, y_train = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        random_state=42,
    )

    # Define a trial object
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    # Call the objective function
    score = optuna_xgb_obj(trial, X_train, y_train, random_state=42)

    # Assertions
    assert isinstance(score, float), "The returned score should be a float."
    print(f"Test passed for optuna_xgb_obj with score: {score}")


def test_optuna_optimize_xgb():
    # Generate synthetic data
    np.random.seed(0)
    X_train, y_train = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        random_state=42,
    )

    # Call the function (using a small number of trials for testing purposes)
    best_model = optuna_optimize_xgb(
        X_train, y_train, random_state=42, n_trials=5
    )

    # Assertions
    assert isinstance(
        best_model, XGBClassifier
    ), "The returned model should be an XGBClassifier."
    print("Test passed for optuna_optimize_xgb.")


def test_make_bins():
    # Generate synthetic predicted probabilities
    y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2])

    # Define bins and labels
    bins = 3
    labels = ["Low", "Medium", "High"]

    # Call the function
    binned = make_bins(y_pred, bins, labels)

    # Assertions
    assert len(binned) == len(
        y_pred
    ), "The length of binned output should match y_pred."
    assert set(binned) <= set(
        labels
    ), "Binned output should only contain specified labels."
    print("Test passed for make_bins.")


def test_fit_cox_model():
    # Generate minimal synthetic data
    np.random.seed(0)
    n_samples = 50

    # Create training data
    train_data = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=10, size=n_samples),
            "event": np.random.binomial(1, 0.7, size=n_samples),
            "feature1": np.random.normal(size=n_samples),
            "feature2": np.random.normal(size=n_samples),
        }
    )

    # Create test data
    test_data = pd.DataFrame(
        {
            "duration": np.random.exponential(scale=10, size=n_samples),
            "event": np.random.binomial(1, 0.7, size=n_samples),
            "feature1": np.random.normal(size=n_samples),
            "feature2": np.random.normal(size=n_samples),
        }
    )

    # Define duration, event columns, and selected features
    duration_col = "duration"
    event_col = "event"
    selected_features = ["feature1", "feature2"]

    # Call the fit_cox_model function
    cph = fit_cox_model(
        train_data, test_data, duration_col, event_col, selected_features
    )

    # Check that the returned object is an instance of CoxPHFitter
    assert isinstance(
        cph, CoxPHFitter
    ), "Returned object is not a CoxPHFitter instance."

    print("fit_cox_model test passed.")
