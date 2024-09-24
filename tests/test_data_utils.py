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
