from __future__ import annotations

import os
import sys

from loguru import logger

# setup logger
logger.remove()
logging_level = os.environ.get("PADDLEFX_LOG_LEVEL", "DEBUG")
fmt = None
fmt = '<level>{level: <8}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
logger.add(sys.stdout, level=logging_level, format=fmt)
