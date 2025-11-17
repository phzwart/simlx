"""Logging helpers for training runs."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO, name: str | None = None) -> logging.Logger:
    """Configure and return a logger with sensible defaults."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
