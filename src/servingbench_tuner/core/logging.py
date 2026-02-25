from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def setup_logging(level: str | None = None) -> None:
    """
    Configure Rich logging. Safe to call multiple times.
    """
    lvl = (level or os.environ.get("SBT_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    # Reduce noise from http libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
