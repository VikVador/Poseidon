r"""A collection of tools for the project."""

import ast

from typing import Dict, Sequence

# isort: split
from poseidon.config import POSEIDON_GRID, POSEIDON_PTRC, PROJECT_FOLDER_DATA


def generate_paths() -> Dict[tuple, Sequence[str]]:
    r"""Generate paths to physical and biogeochemical datasets (1980 to 2022)."""
    with open(POSEIDON_GRID, "r") as file:
        physics_data = ast.literal_eval(file.read())
    with open(POSEIDON_PTRC, "r") as file:
        biogeochemistry_data = ast.literal_eval(file.read())
    dataset_paths = {}
    for date_key in physics_data:
        combined_paths = physics_data[date_key] + biogeochemistry_data[date_key]
        dataset_paths[date_key] = [str(PROJECT_FOLDER_DATA) + path for path in combined_paths]
    return dataset_paths
