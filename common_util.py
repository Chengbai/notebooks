import logging
import json
import torch
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


def top_k_multinomial(probs: torch.tensor, k: int, num_samples: int = 1):
    # Get the top k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

    # Renormalize the top k probabilities
    top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)

    # Sample from the top k probabilities
    samples = torch.multinomial(top_k_probs, num_samples, replacement=True)

    # Map the sampled indices back to the original indices
    return top_k_indices.gather(-1, samples)
