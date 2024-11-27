r"""Tests for the poseidon.data.const module."""

from poseidon.data.const import DATASET_VARIABLES


def test_variables_types():
    assert isinstance(DATASET_VARIABLES, list)
    assert all(isinstance(variable, str) for variable in DATASET_VARIABLES)
