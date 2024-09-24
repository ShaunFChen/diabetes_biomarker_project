import logging
import pytest
from src.utils import *


def test_setup_logging():
    # Test with valid log levels
    for level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        setup_logging(level=level)
        logger = logging.getLogger()
        assert logger.level == getattr(logging, level)

    # Test with invalid log level
    with pytest.raises(ValueError):
        setup_logging(level="INVALID_LEVEL")
