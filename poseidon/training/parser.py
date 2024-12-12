r"""Poseidon - Helper tools for parsing configuration files."""

import yaml

from itertools import product
from pathlib import Path
from typing import Any, Dict, List


def load_configuration(path: Path) -> List[Dict[str, Any]]:
    r"""Load a YAML configuration and expand it into all possible combinations of parameters.

    Arguments:
        path: Path to the .yml configuration file.

    Returns:
        List of dictionaries, each representing a unique combination of the parameters.
    """

    def generate_combinations(d: Dict[str, Any]) -> List[Dict[str, Any]]:
        r"""Recursively generate parameter combinations."""
        if isinstance(d, dict):
            combinations = {k: generate_combinations(v) for k, v in d.items()}
            keys, values = zip(*combinations.items())
            return [dict(zip(keys, combo)) for combo in product(*values)]
        return d if isinstance(d, list) else [d]

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    return generate_combinations(config)
