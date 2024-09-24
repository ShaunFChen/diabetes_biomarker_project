import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import multiprocessing as mp


def calculate_propensity_scores(df, covariates, outcome_col):
    """
    Calculate propensity scores using logistic regression on covariates.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        covariates (list): List of covariate column names.
        outcome_col (str): Name of the outcome column.

    Returns:
        pandas.DataFrame: DataFrame with an added 'propensity_score' column.
    """
    df = df.copy()
    X_covariates = df[covariates]
    y = df[outcome_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_covariates)
    propensity_model = LogisticRegression(solver="liblinear")
    propensity_model.fit(X_scaled, y)
    df["propensity_score"] = propensity_model.predict_proba(X_scaled)[:, 1]
    return df


def match_propensity_scores(df, outcome_col):
    """
    Perform nearest neighbor matching based on propensity scores.

    Args:
        df (pandas.DataFrame): DataFrame containing 'propensity_score' and outcome column.
        outcome_col (str): Name of the outcome column.

    Returns:
        pandas.DataFrame: Matched DataFrame containing both cases and matched controls.
    """
    cases = df[df[outcome_col] == 1].copy()
    controls = df[df[outcome_col] == 0].copy()
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(controls[["propensity_score"]])
    distances, indices = nbrs.kneighbors(cases[["propensity_score"]])
    matched_controls = controls.iloc[indices.flatten()]
    matched_data = pd.concat([cases, matched_controls], ignore_index=True)
    return matched_data


def run_single_logistic_regression(biomarker, df, covariates, outcome_col):
    """
    Run logistic regression on a single biomarker and calculate log2 fold change and p-value.

    Args:
        biomarker (str): Name of the biomarker column.
        df (pandas.DataFrame): DataFrame containing the data.
        covariates (list): List of covariate column names.
        outcome_col (str): Name of the outcome column.

    Returns:
        tuple: A tuple containing log2 fold change and p-value.
    """
    X = df[[biomarker] + covariates]
    X = sm.add_constant(X)
    logit_model = sm.Logit(df[outcome_col], X)
    result = logit_model.fit(disp=0)
    odds_ratio = np.exp(result.params.iloc[1])
    log2_fold_change = np.log2(odds_ratio)
    p_value = result.pvalues.iloc[1]
    return log2_fold_change, p_value


def run_logistic_regression_mp(
    df, covariates, outcome_col, biomarker_cols, n_cores=8
):
    """
    Run logistic regression on multiple biomarkers using multiprocessing.

    Args:
        df (pandas.DataFrame): DataFrame containing the data.
        covariates (list): List of covariate column names.
        outcome_col (str): Name of the outcome column.
        biomarker_cols (list): List of biomarker column names.
        n_cores (int, optional): Number of CPU cores to use. Defaults to 8.

    Returns:
        tuple: Two lists containing log2 fold changes and p-values for each biomarker.
    """
    with mp.Pool(processes=n_cores) as pool:
        results = pool.starmap(
            run_single_logistic_regression,
            [
                (biomarker, df, covariates, outcome_col)
                for biomarker in biomarker_cols
            ],
        )

    log2_fold_changes, p_values = zip(*results)
    return list(log2_fold_changes), list(p_values)


def adjust_p_values(p_values):
    """
    Adjust p-values using the Benjamini-Hochberg procedure.

    Args:
        p_values (list or array-like): List of p-values to adjust.

    Returns:
        array: Corrected p-values.
    """
    _, corrected_p_values, _, _ = multipletests(p_values, method="fdr_bh")
    return corrected_p_values


def filter_highly_correlated_features(
    df, significant_biomarkers, threshold=0.5
):
    """
    Filter out highly correlated features, keeping the one with higher ranking.

    Args:
        df (pandas.DataFrame): DataFrame with features as columns.
        significant_biomarkers (pandas.DataFrame): DataFrame with biomarker ranking information (log2FC and p-value).
        threshold (float): Correlation threshold above which features are considered highly correlated.

    Returns:
        list: List of highly correlated features to be removed.
    """
    # Calculate absolute correlation matrix and mask upper triangle
    corr_matrix = df.corr()
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    high_corr_pairs = corr_matrix.where(mask).stack().reset_index()
    high_corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
    high_corr_pairs = high_corr_pairs[
        high_corr_pairs["Correlation"] > threshold
    ]

    # Keep track of features to drop
    features_to_drop = set()

    # Iterate over high correlation pairs and drop the lower ranking one
    for _, row in high_corr_pairs.iterrows():
        feature_1, feature_2 = row["Feature_1"], row["Feature_2"]

        if feature_1 in features_to_drop or feature_2 in features_to_drop:
            continue

        # Get the log2FC and p-value for both features
        feature_1_info = significant_biomarkers[
            significant_biomarkers["biomarker"] == feature_1
        ]
        feature_2_info = significant_biomarkers[
            significant_biomarkers["biomarker"] == feature_2
        ]

        if not feature_1_info.empty and not feature_2_info.empty:
            # Compare log2FC first, then corrected p-value if log2FCs are equal
            if abs(feature_1_info["log2FC"].values[0]) > abs(
                feature_2_info["log2FC"].values[0]
            ):
                features_to_drop.add(feature_2)
            elif abs(feature_1_info["log2FC"].values[0]) < abs(
                feature_2_info["log2FC"].values[0]
            ):
                features_to_drop.add(feature_1)
            else:
                # If log2FC is the same, compare corrected p-value (lower is better)
                if (
                    feature_1_info["corrected_p_value"].values[0]
                    < feature_2_info["corrected_p_value"].values[0]
                ):
                    features_to_drop.add(feature_2)
                else:
                    features_to_drop.add(feature_1)

    # Return the list with highly correlated features to be removed
    return list(features_to_drop)
