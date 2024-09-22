import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from pathlib import Path  # Add the missing import
from src.data_utils import load_data, iterative_imputation, check_file_exists


# Set seed globally for all tests
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(19770525)


# Test load_data function
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="subject_id,mtb_0018261,BMI,age,sex\n",
)
@patch("src.data_utils.check_file_exists")
@patch("src.data_utils.pd.read_csv")
def test_load_data(mock_read_csv, mock_check_file_exists, mock_file):
    # Arrange
    raw_data_path = "fake_path.csv"
    categorical_columns = ["subject_id", "sex"]
    numeric_columns = ["BMI", "age"]
    biomarker_pattern = r"^mtb_"

    # Fake DataFrame returned by read_csv
    mock_df = pd.DataFrame(
        {
            "subject_id": ["sbj_0000", "sbj_0001"],
            "mtb_0018261": [5885.011, 7624.425],
            "BMI": [18.66, 28.17],
            "age": [33.81, 68.56],
            "sex": ["male", "male"],
        }
    )
    mock_read_csv.return_value = mock_df

    # Act
    df = load_data(
        raw_data_path, categorical_columns, numeric_columns, biomarker_pattern
    )

    # Assert
    mock_check_file_exists.assert_called_once_with(Path(raw_data_path))
    mock_file.assert_called_once_with(raw_data_path, "r")
    mock_read_csv.assert_called_once_with(
        raw_data_path,
        dtype={
            "subject_id": "object",
            "sex": "object",
            "BMI": np.float64,
            "age": np.float64,
            "mtb_0018261": np.float64,
        },
        engine="pyarrow",
    )
    assert df.equals(mock_df)
