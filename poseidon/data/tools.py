r"""A collection of tools designed for data module."""

import ast

from typing import Dict, Sequence

# isort: split
from poseidon.config import (
    PATH_GRID,
    PATH_PTRC,
    SIMULATION_DATA,
)


def generate_paths() -> Dict[str, Sequence[str]]:
    r"""Generate paths to access Black Sea simulation monthly grouped results (1980 to 2022).

    Returns:
        A dictionary where each key is a "YEAR-MONTH" string, and the corresponding
        value is a list of paths to access the simulation data in .netcdf format.
    """

    # Accessing hydrodynamics data
    with open(PATH_GRID, "r") as file:
        physics_data = ast.literal_eval(file.read())

    # Accessing biogeochemistry data
    with open(PATH_PTRC, "r") as file:
        biogeochemistry_data = ast.literal_eval(file.read())

    paths = {}
    for date_month in physics_data:
        # Extracting all relevant paths
        paths_phys_and_bio = physics_data[date_month] + biogeochemistry_data[date_month]

        # Combining them with path to Black Sea simulation folder
        paths[date_month] = [SIMULATION_DATA / p.lstrip("/") for p in paths_phys_and_bio]

    return paths
