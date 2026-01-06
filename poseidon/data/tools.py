r"""A collection of tools designed for data module."""

import ast
import re

from typing import Dict, Sequence

# isort: split
from poseidon.config import (
    PATH_GRID,
    PATH_PTRC,
)


def assert_date_format(date_string: str) -> None:
    r"""Asserts that the date string is in the correct format (YYYY-MM-DD)."""
    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
    if not re.match(pattern, date_string):
        raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")


def generate_paths() -> Dict[str, Sequence[str]]:
    r"""Generate paths to access Black Sea simulation monthly grouped results (1980 to 2022)."""

    with open(PATH_GRID, "r") as file:
        physics_data = ast.literal_eval(file.read())
    with open(PATH_PTRC, "r") as file:
        biogeochemistry_data = ast.literal_eval(file.read())
    paths = {}
    for date_month in physics_data:
        paths[date_month] = physics_data[date_month] + biogeochemistry_data[date_month]

    return paths
