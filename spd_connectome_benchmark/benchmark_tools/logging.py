"""Logging setup shared by command-line benchmark entry points."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging once for a benchmark command."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a package logger with the shared benchmark naming convention."""
    return logging.getLogger(name)
