r"""A collection of tools designed for diagnostics module."""

from typing import Dict

# isort: split
from poseidon.diagnostics.const import DAYS_IN_MONTH


def create_day_index_mapping() -> Dict[str, int]:
    """Creates a mapping from date strings to their corresponding day index."""
    day_index_map = {}
    day_counter = 1

    for month in range(1, 13):
        for day in range(1, DAYS_IN_MONTH[month] + 1):
            key = f"{month:02d}-{day:02d}"
            day_index_map[key] = day_counter
            day_counter += 1

    return day_index_map
