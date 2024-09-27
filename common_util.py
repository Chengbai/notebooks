import json

from pathlib import Path
from typing import Dict


def load_json(json_file: str) -> Dict:
    assert json_file
    assert Path(json_file).exists()
    with open(json_file, "r") as f:
        json_data = json.load(f)

    return json_data
