r"""A collection of tools designed for data module."""

import ast
import numpy as np
import pandas as pd
import re
import torch

from typing import Dict, Sequence

# isort: split
from poseidon.config import (
    PATH_GRID,
    PATH_PTRC,
    SIMULATION_DATA,
)


def assert_date_format(date_string: str) -> None:
    r"""Asserts that the date string is in the correct format (YYYY-MM-DD).

    Arguments:
        date_string (str): Date string to check.

    Raises:
        ValueError: If the date string does not match the pattern.
    """

    # YYYY-MM-DD
    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"

    if not re.match(pattern, date_string):
        raise ValueError("The format is incorrect. The date should be in YYYY-MM-DD format.")


def get_date_features(date: np.datetime64) -> torch.Tensor:
    r"""Extracts year, month, day, and hour from a numpy.datetime64 object.

    Arguments:
        date: Anobject representing a specific date and time.

    Returns:
        A tensor containing the month, day, and hour extracted from the input date.
    """
    timestamp = pd.to_datetime(date)
    return torch.as_tensor([timestamp.year, timestamp.month, timestamp.day, timestamp.hour])


def generate_paths() -> Dict[str, Sequence[str]]:
    r"""Generate paths to access Black Sea simulation monthly grouped results (1980 to 2022).

    Returns:
        A dictionary where each key is a "YEAR-MONTH" string, and the corresponding
        value is a list of paths to access the simulation data in .netcdf format.
    """

    with open(PATH_GRID, "r") as file:
        physics_data = ast.literal_eval(file.read())
    with open(PATH_PTRC, "r") as file:
        biogeochemistry_data = ast.literal_eval(file.read())

    paths = {}
    for date_month in physics_data:
        paths_phys_and_bio = physics_data[date_month] + biogeochemistry_data[date_month]
        paths[date_month] = [SIMULATION_DATA / p.lstrip("/") for p in paths_phys_and_bio]

    return paths
