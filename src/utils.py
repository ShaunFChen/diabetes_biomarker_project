import logging


def setup_logging(level="INFO"):
    """
    Set up logging configuration.

    Args:
        level (str, optional): Logging level as a string (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.

    Returns:
        None
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logger.setLevel(numeric_level)
