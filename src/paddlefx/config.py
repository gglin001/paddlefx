from __future__ import annotations

import os
import sys

from loguru import logger

# setup logger
logging_level = os.environ.get("PADDLEFX_LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=logging_level)
