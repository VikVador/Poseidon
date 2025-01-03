"""Tests for the poseidon.config module."""

import pytest

from pathlib import Path
from poseidon.config import (
    PATH_DATA,
    PATH_GRID,
    PATH_MASK,
    PATH_MASTDB,
    PATH_MESH,
    PATH_MODEL,
    PATH_OBS,
    PATH_OBSERVATIONS_FLOATS,
    PATH_OBSERVATIONS_SATELLITE,
    PATH_PTRC,
    PATH_STAT,
    POSEIDON,
    SCRATCH,
    SIMULATION,
    SIMULATION_DATA,
    SIMULATION_MASK,
)


def test_mask_and_mesh():
    """Test mask and mesh paths for consistency."""
    assert (
        PATH_MASK.suffix == ".zarr"
    ), f"ERROR - PATH_MASK should be a .zarr file but got {PATH_MASK.suffix}."
    assert (
        PATH_MESH.suffix == ".zarr"
    ), f"ERROR - PATH_MESH should be a .zarr file but got {PATH_MESH.suffix}."


def test_paths_are_strings_or_paths():
    """Ensure all paths are instances of Path or None."""
    paths = [
        SIMULATION,
        SIMULATION_DATA,
        SIMULATION_MASK,
        PATH_MASTDB,
        SCRATCH,
        POSEIDON,
        PATH_DATA,
        PATH_OBS,
        PATH_STAT,
        PATH_PTRC,
        PATH_GRID,
        PATH_MASK,
        PATH_MESH,
        PATH_MODEL,
    ]

    for path in paths:
        assert isinstance(path, (Path, type(None))), f"ERROR - {path} is not a Path or None."


@pytest.mark.parametrize("path_dict", [PATH_OBSERVATIONS_FLOATS, PATH_OBSERVATIONS_SATELLITE])
def test_observations_structure(path_dict):
    """Verify that observation structures have expected keys and values."""
    for region, observations in path_dict.items():
        assert region in {"shelf", "black_sea"}, f"ERROR - Unexpected region: {region}."
        for _, entries in observations.items():
            if isinstance(entries, dict):
                for level, path in entries.items():
                    assert level in {"L3", "L4"}, f"ERROR - Unexpected level: {level}."
                    assert path is None or isinstance(
                        path, Path
                    ), f"ERROR - {path} is not a valid Path or None."
            else:
                assert isinstance(entries, Path), f"ERROR - {entries} is not a valid Path."
