"""Logging configuration shared across the package and worker processes.

Per project conventions: the formatter includes asctime, levelname, name, lineno
and message; timestamps use the ``%Y-%m-%d %H:%M:%S`` format in the Bangkok timezone.
"""

import logging
import os
import time
from typing import (Optional)

LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the project-standard format and timezone.

    Safe to call from each worker process; ``force=True`` resets any inherited
    handlers so child processes log consistently.
    """
    os.environ.setdefault("TZ", "Asia/Bangkok")
    # ``time.tzset`` is POSIX-only; the project targets macOS/Linux.
    if hasattr(time, "tzset"):
        time.tzset()

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
