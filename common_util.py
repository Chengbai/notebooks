import logging
import json
import sys

from pathlib import Path
from typing import Dict

loggers = {}


def get_logger(name: str):
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        _logger = logging.getLogger(name)
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stdout.setFormatter(formatter)
        _logger.addHandler(stdout)
        _logger.setLevel(logging.INFO)

        loggers[name] = _logger
    return _logger


def load_json(json_file: str) -> Dict:
    assert json_file
    assert Path(json_file).exists()
    with open(json_file, "r") as f:
        json_data = json.load(f)

    return json_data
