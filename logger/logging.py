from __future__ import annotations

"""Simple reusable logger configuration for Weekend Wizard."""

import logging
import sys


def get_logger(name: str, *_, **__) -> logging.Logger:
    """Create and return a configured logger instance."""
    if not name.startswith("weekend_wizard."):
        name = f"weekend_wizard.{name}"

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # MCP stdio uses stdout for JSON-RPC, so logs must stay on stderr.
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
