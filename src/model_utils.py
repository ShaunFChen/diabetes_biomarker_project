import numpy as np
import pandas as pd
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
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt


def prepare_data(df, outcome_column, exclude_columns=None):
    """
    Prepare the feature matrix X and target vector y by excluding specified columns.

    Args:
        df (pandas.DataFrame): Original DataFrame.
        outcome_column (str): The name of the outcome column.
        exclude_columns (list, optional): List of columns to exclude from features.

    Returns:
        tuple: A tuple (X, y) where X is the feature matrix and y is the target vector.
    """
    # Handle case where exclude_columns is None
    if exclude_columns is None:
        exclude_columns = []

    X = df.drop(columns=set(exclude_columns + [outcome_column]))
    y = df[outcome_column]
    return X, y


def get_prediction_probabilities(models, X_train, y_train, X_test):
    """
    Train models and return the predicted probabilities for the test set.

    Args:
        models (dict): Dictionary of model names and model objects.
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target vector.
        X_test (array-like): Test feature matrix.

    Returns:
        dict: Dictionary with model names as keys and predicted probabilities as values.
    """
    prediction_probs = {}

    for model_name, model in models.items():
        # Create pipeline with scaling and model
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Standardize features
                ("model", model),  # The actual model
            ]
        )

        # Train the model
        pipeline.fit(X_train, y_train)

        # Save the predicted probabilities for the test set
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        prediction_probs[model_name] = y_pred_prob

    return prediction_probs


def exec_basic_logreg(X_train, y_train, random_state=19770525):
    """
    Execute basic logistic regression training and plot feature importance.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target vector.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        Pipeline: Trained logistic regression pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Standardize the features
            (
                "logreg",
                LogisticRegression(
                    solver="liblinear", random_state=random_state
                ),
            ),  # Logistic regression model
        ]
    )

    # Step 4: Train the model
    pipeline.fit(X_train, y_train)

    # Feature Importance
    logreg_model = pipeline.named_steps[
        "logreg"
    ]  # Extract the trained logistic regression model
    feature_importance = np.abs(
        logreg_model.coef_[0]
    )  # Get the absolute values of the coefficients

    plot_feature_importance(
        X_train.columns,
        feature_importance,
        importance_metric="Coefficient Magnitude",
        title="Feature Importance in Logistic Regression",
        top_N=len(X_train.columns),
    )
    return pipeline


def evaluate_model_performance(PIP, X_test, y_test):
    """
    Evaluate the performance of a classification model and print the results.

    Args:
        PIP (Pipeline): Trained model pipeline.
        X_test (array-like): Test feature matrix.
        y_test (array-like): True labels for test data.

    Returns:
        array: Predicted probabilities for the positive class.
    """
    # Validate that y_test contains only 0 and 1
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("y_test contains labels other than 0 and 1.")

    y_pred = PIP.predict(X_test)
    y_pred_prob = PIP.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC: {auc:.3f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred_prob


def optimize_linear_model(
    X_train,
    y_train,
    param_grid,
    cv,
    estimator,
    scoring="f1_score",
    random_state=19770525,
):
    """
    Perform RFECV feature selection and hyperparameter tuning with GridSearchCV.

    Args:
        X_train (pandas.DataFrame): Training feature matrix.
        y_train (array-like): Training labels.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        cv (cross-validation generator): Cross-validation method.
        estimator (estimator object): Estimator to use (e.g., LogisticRegression).
        scoring (str, optional): Strategy to evaluate the performance of the cross-validated model on the test set.
                                  See options at https://scikit-learn.org/dev/modules/model_evaluation.html#scoring-parameter
        random_state (int, optional): Random state for reproducibility.

    Returns:
        tuple: A tuple containing the best pipeline, selected features, and best hyperparameters.
    """
    rfe_cv_estimator = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        random_state=random_state,
        max_iter=10000,
    )

    # Step 1: Feature selection with RFECV (outside of the pipeline)
    rfecv = RFECV(
        estimator=rfe_cv_estimator,
        step=1,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    # Standardize the features before RFECV
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply RFECV for feature selection
    rfecv.fit(X_train_scaled, y_train)

    # Get the selected features after RFECV
    selected_features = list(X_train.columns[rfecv.support_])
    print(f"Selected features after RFECV: {list(selected_features)}")

    # Step 2: Create a new pipeline for scaling and logistic regression without feature selection
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Standardize the features
            ("estimator", estimator),  # Logistic regression with L1 penalty
        ]
    )

    # Use only the selected features from RFECV for training the model
    X_train_selected = X_train[selected_features]

    # Step 3: Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1,
    )

    # Train the model with selected features
    grid_search.fit(X_train_selected, y_train)

    best_hyperparams = grid_search.best_params_
    best_pipeline = grid_search.best_estimator_

    # Step 4: Display the best hyperparameters
    print(f"Best hyperparameters: {best_hyperparams}")

    return best_pipeline, selected_features, best_hyperparams


def optuna_xgb_obj(trial, X_train, y_train, random_state):
    """
    Objective function for Optuna hyperparameter optimization of XGBoost.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random state for reproducibility.

    Returns:
        float: Mean cross-validated AUC score.
    """
    # Define hyperparameter search space
    params = {
        "objective": trial.suggest_categorical(
            "objective", ["binary:logistic"]
        ),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
        "gamma": trial.suggest_float("gamma", 0, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        "max_bin": trial.suggest_int("max_bin", 2, 20),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 12),
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.05, 20),
        "n_jobs": -1,
        "random_state": random_state,
    }

    # XGBoost classifier with hyperparameters
    xgb_clf = XGBClassifier(**params)

    # Set up pipeline
    xgb_PIP = Pipeline([("XGBoost", xgb_clf)])

    # Cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        xgb_PIP, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
    )

    # Calculate mean cross-validated AUC score
    mean_auc = np.mean(cv_scores)

    # Report intermediate results to Optuna (required for pruning)
    trial.report(mean_auc, step=0)

    # Check if the trial should be pruned
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # Return the mean AUC score
    return mean_auc


def optuna_optimize_xgb(X_train, y_train, random_state, n_trials=100):
    """
    Optimize XGBoost hyperparameters using Optuna and return the best model.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training labels.
        random_state (int): Random state for reproducibility.
        n_trials (int, optional): Number of trials for optimization.

    Returns:
        XGBClassifier: Trained XGBoost model with optimized hyperparameters.
    """
    # Create an Optuna study
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=random_state)
    )

    # Run the optimization with the specified number of trials
    study.optimize(
        lambda trial: optuna_xgb_obj(trial, X_train, y_train, random_state),
        n_trials=n_trials,
    )

    best_params = study.best_trial.params
    print(f"Best hyperparameters: {best_params}")

    # Create the XGBClassifier using the best parameters
    best_xgb_PIP = XGBClassifier(**best_params)

    # Train the best model on the entire training set
    best_xgb_PIP.fit(X_train, y_train)

    return best_xgb_PIP


def make_bins(y_pred, bins, labels):
    """
    Create bins for predicted probabilities using quantile-based discretization.

    Args:
        y_pred (array-like): Predicted probabilities or scores.
        bins (int): Number of bins to split the data into.
        labels (list): Labels for the bins.

    Returns:
        numpy.ndarray: Array of bin labels corresponding to each prediction.
    """
    return pd.qcut(
        pd.Series(y_pred).rank(method="first"), q=bins, labels=labels
    ).values


def fit_cox_model(train_data, test_data, duration_col, event_col, selected_features):
    """
    Fits a Cox Proportional Hazards model to the training data and evaluates it on the test data.

    Args:
        train_data (DataFrame): Training dataset that includes the duration, event, and selected features.
        test_data (DataFrame): Test dataset that includes the duration, event, and selected features.
        duration_col (str): The name of the column representing the follow-up time (duration).
        event_col (str): The name of the column representing the event (1 if event occurred, 0 if censored).
        selected_features (list): List of selected feature names to include in the model (should not include event or duration).

    Returns:
        float: The concordance index (C-index) for the test set.
        CoxPHFitter: The fitted Cox proportional hazards model.
    """
    # Ensure the selected features do not include the duration or event columns
    survival_cols = list(selected_features)

    # Initialize the Cox Proportional Hazards model
    cph = CoxPHFitter()

    # Fit the Cox model on the training data, using the selected features along with duration and event
    cph.fit(
        train_data[survival_cols + [duration_col, event_col]], 
        duration_col=duration_col, 
        event_col=event_col
    )

    # Predict partial hazard ratios for the test data using only the selected features
    predicted_hr = cph.predict_partial_hazard(test_data[survival_cols])

    # Plot the model coefficients
    cph.plot()
    plt.title("Cox Proportional Hazards Model Coefficients")
    plt.show()

    # Calculate the concordance index (C-index) for the test set
    c_index = concordance_index(
        test_data[duration_col],  # Follow-up time in the test set
        -predicted_hr,            # Negative partial hazard ratios for the test set
        test_data[event_col]       # Event indicator for the test set
    )

    print(f"Concordance Index: {c_index:.3f}")

    # Return the C-index and the fitted Cox model
    return cph