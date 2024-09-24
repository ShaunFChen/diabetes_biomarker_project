import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path  # For Python < 3.9
from typing import List

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def check_file_exists(file_path):
    """
    Check if the file exists and raise an error if it does not.

    Args:
        file_path (pathlib.Path): Path to the file.

    Returns:
        None
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}.")


def get_dtypes(categorical_columns, numeric_columns, biomarker_cols):
    """
    Define the data types for each column in the dataset.

    Args:
        categorical_columns (list): List of categorical column names.
        numeric_columns (list): List of numeric column names.
        biomarker_cols (list): List of biomarker column names.

    Returns:
        dict: Dictionary mapping column names to data types.
    """
    dtypes = {col: "object" for col in categorical_columns}
    dtypes.update({col: np.float64 for col in numeric_columns})
    for col in biomarker_cols:
        dtypes[col] = np.float64
    return dtypes


def load_data(
    raw_data_path: str,
    id_column: str,
    categorical_columns: List[str],
    numeric_columns: List[str],
    biomarker_pattern: str,
    engine: str = "c",  # Allow users to specify the engine
) -> pd.DataFrame:
    """
    Load dataset based on column patterns and types.

    Args:
        raw_data_path (str): Path to the raw data CSV file.
        id_column (str): The column to use as the row index.
        categorical_columns (List[str]): List of categorical column names.
        numeric_columns (List[str]): List of numeric column names.
        biomarker_pattern (str): Regex pattern to identify biomarker columns.
        engine (str, optional): Parser engine to use ('c', 'python', or 'pyarrow'). Defaults to 'c'.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    # Check if the file exists
    check_file_exists(Path(raw_data_path))

    # Read only the headers (column names) using readline for speed
    with open(raw_data_path, "r") as file:
        first_line = file.readline().strip()

    # Split the first line by comma (or the appropriate delimiter)
    column_names = first_line.split(",")

    # Create a list of biomarker columns based on the pattern
    biomarker_cols = [
        col for col in column_names if re.search(biomarker_pattern, col)
    ]

    # Define data types for the dataset
    dtypes = get_dtypes(categorical_columns, numeric_columns, biomarker_cols)

    # Load the full dataset with the specified data types
    if engine == "pyarrow":
        # Read without index_col and dtype
        df = pd.read_csv(raw_data_path, engine=engine)
        # Set data types
        for col, dtype in dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        # Set index column
        if id_column in df.columns:
            df.set_index(id_column, inplace=True)
        else:
            raise ValueError(
                f"Index column '{id_column}' not found in the data."
            )
    else:
        # For 'c' or 'python' engines, can use index_col and dtype directly
        df = pd.read_csv(
            raw_data_path, dtype=dtypes, engine=engine, index_col=id_column
        )

    logging.info(f"Dataset shape: {df.shape}")

    return df


def iterative_imputation(
    df,
    missing_threshold=0.3,
    exclude_cols=None,
    estimator=None,
    imputer_kwargs=None,
):
    """
    Perform iterative imputation on a DataFrame, with options to set an estimator, exclude columns, and reattach columns after imputation.

    Args:
        df (pandas.DataFrame): Original DataFrame to be imputed.
        missing_threshold (float, optional): Missingness threshold for dropping columns. Defaults to 0.3.
        exclude_cols (list, optional): Columns to exclude from imputation. Defaults to None.
        estimator (object, optional): Estimator for imputation. Defaults to None.
        imputer_kwargs (dict, optional): Additional keyword arguments for IterativeImputer.

    Returns:
        pandas.DataFrame: The imputed DataFrame with excluded columns reattached.
    """
    # Step 1: Drop columns with more than the threshold of missing values
    cols_to_drop = df.columns[df.isnull().mean() > missing_threshold]
    cleaned_df = df.drop(cols_to_drop, axis=1).copy()

    # Step 2: Set default excluded columns if not provided
    if exclude_cols is None:
        exclude_cols = [
            "subject_id",
            "incident_diabetes",
            "diabetes_followup_time",
        ]

    # Select columns for imputation (excluding specific columns)
    cols_for_imputation = [
        col for col in cleaned_df.columns if col not in exclude_cols
    ]

    # Prepare dataframe for imputation
    for_imputation_df = cleaned_df[cols_for_imputation].copy()

    # Step 3: If no custom estimator is provided, default to IterativeImputer's default
    if estimator is None:
        estimator = "default"

    # Step 4: Handle kwargs for IterativeImputer (set defaults if not passed)
    if imputer_kwargs is None:
        imputer_kwargs = {}

    if estimator == "default":
        # If no special estimator, just use IterativeImputer's default
        imputer = IterativeImputer(**imputer_kwargs)
    else:
        # If a special estimator like RandomForestRegressor is provided
        imputer = IterativeImputer(estimator=estimator, **imputer_kwargs)

    # Step 5: Fit and transform the imputer on the data
    imputed_data = imputer.fit_transform(for_imputation_df)

    # Step 6: Convert the imputed data back to a DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=cols_for_imputation)

    # Step 7: Reattach the excluded columns back to the imputed DataFrame
    for col in exclude_cols:
        if col in df.columns:
            imputed_df[col] = df[col].values

    # Ensure subject_id is in the first column
    cols = ["subject_id"] + [
        col for col in imputed_df.columns if col != "subject_id"
    ]
    imputed_df = imputed_df[cols]

    return imputed_df


def impute_data(
    imputed_file_path,
    transformed_df,
    missing_threshold,
    exclude_cols,
    imputer_kwargs,
    integer_cols,
    estimator=None,
):
    """
    Load imputed data if available, otherwise perform imputation.

    Args:
        imputed_file_path (str): Path to save/load the imputed data.
        transformed_df (pandas.DataFrame): DataFrame to perform imputation on.
        missing_threshold (float): Missingness threshold for dropping columns.
        exclude_cols (list): Columns to exclude from imputation and reattach after imputation.
        imputer_kwargs (dict): Keyword arguments for the imputation process.
        integer_cols (list): Columns that should be rounded to integers.
        estimator (object, optional): Estimator to be used for imputation
                                      (e.g., RandomForestRegressor). Defaults to None.

    Returns:
        pandas.DataFrame: The imputed dataset.
    """
    # Check if the imputed file already exists
    if os.path.exists(imputed_file_path):
        logging.info(
            f"{imputed_file_path}: File exists. Loading imputed data..."
        )
        X_imputed_df = pd.read_pickle(imputed_file_path)
    else:
        logging.info(
            f"{imputed_file_path}: File not found. Performing imputation..."
        )

        # Save a copy of the excluded columns for reattachment
        excluded_cols_df = transformed_df[exclude_cols].copy()

        # Drop columns with more than `missing_threshold` missing values
        cols_to_keep = transformed_df.columns[
            transformed_df.isnull().mean() < missing_threshold
        ]
        transformed_df = transformed_df[cols_to_keep]

        # Exclude specific columns from imputation
        transformed_df = transformed_df.drop(columns=exclude_cols)

        # Perform imputation using IterativeImputer with the custom or default estimator
        imputer = IterativeImputer(estimator=estimator, **imputer_kwargs)
        X_imputed_df = pd.DataFrame(
            imputer.fit_transform(transformed_df),
            columns=transformed_df.columns,
        )

        # Update the index of the imputed DataFrame to match the original DataFrame's index
        X_imputed_df.index = transformed_df.index

        # Convert specified integer columns to integers
        for col in integer_cols:
            if col in X_imputed_df:
                X_imputed_df[col] = np.round(X_imputed_df[col])

        # Reattach the excluded columns back to the imputed DataFrame
        X_imputed_df = pd.concat([X_imputed_df, excluded_cols_df], axis=1)

        # Save the imputed DataFrame
        X_imputed_df.to_pickle(imputed_file_path)

    return X_imputed_df


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
