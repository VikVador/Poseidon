r"""A collection of tools designed for training module."""

import yaml

from itertools import product
from pathlib import Path
from typing import Any, Dict, List


def load_configuration(path: Path) -> List[Dict[str, Any]]:
    r"""Load all combinations of parameters from a YAML configuration file."""

    def generate_combinations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
        r"""Recursively generate parameter combinations."""
        if isinstance(d, dict):
            combinations = {k: generate_combinations(v) for k, v in d.items()}
            keys, values = zip(*combinations.items())
            return [dict(zip(keys, combo)) for combo in product(*values)]
        return d if isinstance(d, list) else [d]

    # Open and read the YAML configuration file
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    # Generate combinations
    return generate_combinations(config)
