"""Shared logging configuration for all harnesses.

One base config, applied once per process entrypoint. Logs carry a timestamp,
level, and harness name so runs are auditable and greppable — important for a
research artifact where every attempt must be traceable.
"""

import logging

from src.shared.constants import LOG_FORMAT, LOG_LEVEL


def configure_logging() -> None:
    """Apply the base logging config. Call once at process start."""
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (typically the harness name)."""
    return logging.getLogger(name)
