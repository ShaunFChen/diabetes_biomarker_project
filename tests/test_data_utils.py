import pytest

import os
import re
import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path  # For Python < 3.9
from typing import List

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.data_utils import *


def test_check_file_exists(tmp_path):
    # Create a temporary file
    temp_file = tmp_path / "temp_file.txt"
    temp_file.touch()  # This creates the file

    # Test that no exception is raised when the file exists
    try:
        check_file_exists(temp_file)
    except FileNotFoundError:
        pytest.fail("check_file_exists raised FileNotFoundError unexpectedly!")

    # Test that FileNotFoundError is raised when the file does not exist
    non_existent_file = tmp_path / "non_existent_file.txt"
    with pytest.raises(FileNotFoundError):
        check_file_exists(non_existent_file)


def test_get_dtypes():
    categorical_columns = ["cat1", "cat2"]
    numeric_columns = ["num1", "num2"]
    biomarker_cols = ["bio1", "bio2"]

    expected_dtypes = {
        "cat1": "object",
        "cat2": "object",
        "num1": np.float64,
        "num2": np.float64,
        "bio1": np.float64,
        "bio2": np.float64,
    }

    result = get_dtypes(categorical_columns, numeric_columns, biomarker_cols)
    assert result == expected_dtypes

    # Test with overlapping columns
    overlapping_cols = ["overlap"]
    categorical_columns += overlapping_cols
    numeric_columns += overlapping_cols
    biomarker_cols += overlapping_cols

    result = get_dtypes(categorical_columns, numeric_columns, biomarker_cols)
    assert result["overlap"] == np.float64


@pytest.mark.parametrize("engine", ["c", "pyarrow"])
def test_load_data(tmp_path, engine):
    # Create sample data
    data = {
        "id": [1, 2, 3],
        "cat1": ["a", "b", "c"],
        "num1": [1.1, 2.2, 3.3],
        "bio_abc": [0.1, 0.2, 0.3],
        "bio_def": [0.4, 0.5, 0.6],
    }
    df = pd.DataFrame(data)
    raw_data_path = tmp_path / "test_data.csv"
    df.to_csv(raw_data_path, index=False)

    id_column = "id"
    categorical_columns = ["cat1"]
    numeric_columns = ["num1"]
    biomarker_pattern = r"^bio_.*"

    loaded_df = load_data(
        raw_data_path=str(raw_data_path),
        id_column=id_column,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        biomarker_pattern=biomarker_pattern,
        engine=engine,
    )

    # Check that the DataFrame is loaded correctly
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (3, 4)
    assert list(loaded_df.columns) == ["cat1", "num1", "bio_abc", "bio_def"]
    assert loaded_df.index.name == "id"

    # Check that dtypes are correct
    expected_dtypes = {
        "cat1": "object",
        "num1": np.float64,
        "bio_abc": np.float64,
        "bio_def": np.float64,
    }
    actual_dtypes = {col: loaded_df[col].dtype for col in loaded_df.columns}
    for col, dtype in expected_dtypes.items():
        assert actual_dtypes[col] == dtype


def test_iterative_imputation():
    # Create a sample DataFrame with missing values
    data = {
        "subject_id": [1, 2, 3, 4, 5],
        "A": [1, 2, np.nan, 4, 5],
        "B": [5, np.nan, np.nan, 8, 10],
        "C": [np.nan, 1, 2, 3, 4],
        "incident_diabetes": [0, 1, 0, 1, 0],
        "diabetes_followup_time": [10, 15, 20, 25, 30],
    }
    df = pd.DataFrame(data)

    # Perform imputation
    imputed_df = iterative_imputation(
        df,
        missing_threshold=0.3,
        exclude_cols=[
            "subject_id",
            "incident_diabetes",
            "diabetes_followup_time",
        ],
        estimator=None,
        imputer_kwargs={"random_state": 0},
    )

    # Determine expected imputed columns
    expected_imputed_cols = ["A", "C"]  # 'B' is dropped

    # Check that missing values have been imputed in expected columns
    assert imputed_df[expected_imputed_cols].isnull().sum().sum() == 0

    # Check that 'B' is not in the imputed DataFrame
    assert "B" not in imputed_df.columns

    # Check that excluded columns are present
    for col in ["subject_id", "incident_diabetes", "diabetes_followup_time"]:
        assert col in imputed_df.columns
        assert imputed_df[col].equals(df[col])

    # Check that columns with missingness over the threshold have been dropped
    cols_dropped = df.columns[df.isnull().mean() > 0.3]
    for col in cols_dropped:
        assert col not in imputed_df.columns


def test_impute_data(tmp_path):
    # Create a sample DataFrame
    data = {
        "id": [1, 2, 3, 4, 5],
        "A": [1, 2, np.nan, 4, 5],
        "B": [5, np.nan, np.nan, 8, 10],
        "C": [np.nan, 1, 2, 3, 4],
        "D": [10, 20, 30, 40, np.nan],  # Integer column
        "exclude1": ["x", "y", "z", "w", "v"],
    }
    transformed_df = pd.DataFrame(data).set_index("id")

    imputed_file_path = tmp_path / "imputed_data.pkl"
    missing_threshold = 0.3
    exclude_cols = ["exclude1"]
    imputer_kwargs = {"random_state": 0}
    integer_cols = ["D"]
    estimator = None

    # Ensure the imputed file does not exist
    assert not imputed_file_path.exists()

    # Perform imputation
    imputed_df = impute_data(
        imputed_file_path=str(imputed_file_path),
        transformed_df=transformed_df,
        missing_threshold=missing_threshold,
        exclude_cols=exclude_cols,
        imputer_kwargs=imputer_kwargs,
        integer_cols=integer_cols,
        estimator=estimator,
    )

    # Determine expected imputed columns (excluding those dropped)
    expected_imputed_cols = ["A", "C", "D"]  # 'B' is dropped

    # Check that the imputed DataFrame has no missing values in expected columns
    assert imputed_df[expected_imputed_cols].isnull().sum().sum() == 0

    # Check that 'B' is not in the imputed DataFrame
    assert "B" not in imputed_df.columns

    # Check that integer columns are rounded
    assert (
        imputed_df["D"].dtype == float
    )  # Note: remains float unless explicitly converted
    fractional_parts = imputed_df["D"] % 1
    assert np.all((fractional_parts == 0) | (imputed_df["D"].isnull()))

    # Check that excluded columns are reattached
    for col in exclude_cols:
        assert col in imputed_df.columns
        assert imputed_df[col].equals(transformed_df[col])

    # Check that the imputed file was saved
    assert imputed_file_path.exists()

    # Modify the saved imputed file to test loading
    imputed_df_saved = pd.read_pickle(imputed_file_path)
    imputed_df_saved["A"] = 999  # Modify a value
    imputed_df_saved.to_pickle(imputed_file_path)

    # Load the data again
    imputed_df_loaded = impute_data(
        imputed_file_path=str(imputed_file_path),
        transformed_df=transformed_df,
        missing_threshold=missing_threshold,
        exclude_cols=exclude_cols,
        imputer_kwargs=imputer_kwargs,
        integer_cols=integer_cols,
        estimator=estimator,
    )

    # Ensure that the loaded data matches the modified data
    assert imputed_df_loaded.equals(imputed_df_saved)
