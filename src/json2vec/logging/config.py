import json
import os
import sys

from loguru import logger
from rich.console import Console
from rich.json import JSON

console = Console(file=sys.stdout)

LOG_LEVEL: str = os.getenv("JSON2VEC_LOG_LEVEL", "DEBUG").upper()

def sink(message):
    record = message.record
    extras = {k: str(v) for k, v in record["extra"].items()}
    payload = {"level": record["level"].name, **extras, "message": record["message"]}
    console.print(JSON(json.dumps(payload), indent=None))


logger.remove()
logger.add(sink=sink, level=LOG_LEVEL, enqueue=True, backtrace=True, diagnose=False)
logger.bind(component="logging", level=LOG_LEVEL).info("configured loguru sink")
