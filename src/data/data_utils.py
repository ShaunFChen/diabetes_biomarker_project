import re
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

def setup_logging(level="INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            level=numeric_level, 
            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logger.setLevel(numeric_level)

def check_file_exists(file_path):
    """Raise an error if the file does not exist."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}.")

def get_dtypes(category_columns, numeric_columns, biomarker_cols):
    """
    Define the data types for each column in the dataset.
    
    Parameters:
    category_columns (list): List of categorical column names
    numeric_columns (list): List of numeric column names
    biomarker_cols (list): List of biomarker column names

    Returns:
    dict: A dictionary mapping column names to data types
    """
    dtypes = {col: 'category' for col in category_columns}
    dtypes.update({col: np.float64 for col in numeric_columns})
    for col in biomarker_cols:
        dtypes[col] = np.float64
    return dtypes

def load_data(
    raw_data_path: str,
    category_columns: List[str],
    numeric_columns: List[str],
    biomarker_pattern: str
) -> pd.DataFrame:
    """
    Load dataset based on column patterns and types.
    
    Parameters:
    raw_data_path (str): Path to the raw data CSV file
    category_columns (list): List of categorical column names
    numeric_columns (list): List of numeric column names
    biomarker_pattern (str): Regex pattern to identify biomarker columns

    Returns:
    pd.DataFrame: Loaded dataset
    """
    # Check if the file exists
    check_file_exists(Path(raw_data_path))

    # Read only the headers (column names) using readline for speed
    with open(raw_data_path, 'r') as file:
        first_line = file.readline().strip()

    # Split the first line by comma (or the appropriate delimiter)
    column_names = first_line.split(',')

    # Create a list of biomarker columns based on the pattern
    biomarker_cols = [
        col for col in column_names 
        if re.search(biomarker_pattern, col)
    ]

    # Define data types for the dataset
    dtypes = get_dtypes(category_columns, numeric_columns, biomarker_cols)

    # Load the full dataset with the specified data types
    raw_df = pd.read_csv(raw_data_path, 
                         # dtype=dtypes, 
                         engine='pyarrow')

    logging.info(f"Dataset shape: {raw_df.shape}")

    return raw_df
