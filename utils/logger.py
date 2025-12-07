import logging
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def get_logger(name: str, level: Optional[str] = "INFO") -> logging.Logger:
    """
    Return a configured logger that all pipeline/scoring modules use.
    Ensures consistent formatting across the system.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is reused
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
