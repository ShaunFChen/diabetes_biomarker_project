import pytest

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment

from src.analysis_utils import *


def test_calculate_propensity_scores():
    # Create a sample DataFrame
    data = {
        'cov1': [1, 2, 3, 4, 5],
        'cov2': [5, 4, 3, 2, 1],
        'outcome': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    covariates = ['cov1', 'cov2']
    outcome_col = 'outcome'
    
    # Run the function
    result_df = calculate_propensity_scores(df, covariates, outcome_col)
    
    # Check that 'propensity_score' column is added
    assert 'propensity_score' in result_df.columns
    
    # Check that propensity scores are between 0 and 1
    assert result_df['propensity_score'].between(0, 1).all()
    
    # Check that the original DataFrame is not modified
    assert 'propensity_score' not in df.columns


def test_match_with_replacement():
    # Create sample data with sufficient controls
    data = {
        'propensity_score': [0.2, 0.4, 0.6, 0.8],
        'outcome': [0, 0, 1, 1],
        'other_feature': [10, 20, 30, 40]  # Additional column to check data integrity
    }
    df = pd.DataFrame(data)
    outcome_col = 'outcome'

    # Perform matching with replacement
    matched_df = match_propensity_scores(df, outcome_col, replace=True)

    # Verify that the matched DataFrame has the correct number of rows
    assert len(matched_df) == 4  # 2 cases + 2 controls

    # Verify that the number of cases and controls is equal
    counts = matched_df[outcome_col].value_counts()
    assert counts[0] == 2  # Controls
    assert counts[1] == 2  # Cases

    # Ensure that all original columns are present
    assert set(matched_df.columns) == set(df.columns)

    # Optional: Check that controls may be matched multiple times
    # Since we have sufficient controls, they might not be matched multiple times


def test_match_without_replacement():
    # Create sample data with sufficient controls
    data = {
        'propensity_score': [0.1, 0.3, 0.5, 0.7],
        'outcome': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    outcome_col = 'outcome'

    # Perform matching without replacement
    matched_df = match_propensity_scores(df, outcome_col, replace=False)

    # Verify that the matched DataFrame has the correct number of rows
    assert len(matched_df) == 4  # 2 cases + 2 controls

    # Verify that the number of cases and controls is equal
    counts = matched_df[outcome_col].value_counts()
    assert counts[0] == 2  # Controls
    assert counts[1] == 2  # Cases

    # Ensure that controls are matched only once
    controls = matched_df[matched_df[outcome_col] == 0]
    assert controls.index.is_unique


def test_match_without_replacement_insufficient_controls():
    # Create sample data with more cases than controls
    data = {
        'propensity_score': [0.2, 0.4, 0.6],
        'outcome': [0, 1, 1]
    }
    df = pd.DataFrame(data)
    outcome_col = 'outcome'

    # Perform matching without replacement
    matched_df = match_propensity_scores(df, outcome_col, replace=False)

    # Verify that the original DataFrame is returned
    pd.testing.assert_frame_equal(matched_df.reset_index(drop=True), df.reset_index(drop=True))






def test_run_single_logistic_regression():
    # Create a sample DataFrame
    data = {
        'biomarker1': [1, 2, 3, 4, 5],
        'cov1': [5, 4, 3, 2, 1],
        'outcome': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    biomarker = 'biomarker1'
    covariates = ['cov1']
    outcome_col = 'outcome'
    
    # Run the function
    log2_fc, p_value = run_single_logistic_regression(biomarker, df, covariates, outcome_col)
    
    # Check that log2_fc and p_value are numeric
    assert isinstance(log2_fc, float)
    assert isinstance(p_value, float)
    
    # Check that p_value is between 0 and 1
    assert 0 <= p_value <= 1


def test_run_logistic_regression_mp():
    # Create a sample DataFrame with multiple biomarkers
    data = {
        'biomarker1': [1, 2, 3, 4, 5],
        'biomarker2': [5, 4, 3, 2, 1],
        'cov1': [2, 2, 2, 2, 2],
        'outcome': [0, 0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    covariates = ['cov1']
    outcome_col = 'outcome'
    biomarker_cols = ['biomarker1', 'biomarker2']
    
    # Run the function
    log2_fcs, p_values = run_logistic_regression_mp(df, covariates, outcome_col, biomarker_cols, n_cores=2)
    
    # Check that the outputs are lists of correct length
    assert isinstance(log2_fcs, list)
    assert isinstance(p_values, list)
    assert len(log2_fcs) == len(biomarker_cols)
    assert len(p_values) == len(biomarker_cols)
    
    # Check that p-values are between 0 and 1
    assert all(0 <= p <= 1 for p in p_values)


def test_adjust_p_values():
    # Create a list of p-values
    p_values = [0.01, 0.04, 0.03, 0.20, 0.50]
    
    # Run the function
    corrected_p_values = adjust_p_values(p_values)
    
    # Check that corrected p-values are the same length
    assert len(corrected_p_values) == len(p_values)
    
    # Check that corrected p-values are between 0 and 1
    assert np.all((corrected_p_values >= 0) & (corrected_p_values <= 1))
    
    # Check that corrected p-values are sorted in the same order
    # (since Benjamini-Hochberg adjusts but maintains order)
    assert np.all(np.diff(corrected_p_values) >= 0)


def test_filter_highly_correlated_features():
    # Create a DataFrame with highly correlated features
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],  # Perfect correlation with feature1
        'feature3': [5, 3, 6, 2, 1],
        'feature4': [5, 3, 6, 2, 1],   # Perfect correlation with feature3
    }
    df = pd.DataFrame(data)
    
    # Create significant_biomarkers DataFrame
    significant_biomarkers = pd.DataFrame({
        'biomarker': ['feature1', 'feature2', 'feature3', 'feature4'],
        'log2FC': [1.5, 2.0, 0.5, 1.0],
        'corrected_p_value': [0.01, 0.02, 0.03, 0.04]
    })
    
    # Run the function
    features_to_drop = filter_highly_correlated_features(df, significant_biomarkers, threshold=0.9)
    
    # Since feature1 and feature2 are perfectly correlated,
    # and feature2 has higher log2FC, feature1 should be dropped
    # Similarly for feature3 and feature4
    assert set(features_to_drop) == {'feature1', 'feature3'}


