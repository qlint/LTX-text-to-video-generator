from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structured logs for API and worker processes."""

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

