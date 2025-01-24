r"""Tests for the poseidon.data.const module."""

import pytest

# isort: split
from poseidon.data.const import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_REGION,
    DATASET_VARIABLES,
    DATASET_VARIABLES_ATMOSPHERE,
    DATASET_VARIABLES_SURFACE,
    TOY_DATASET_DATES_TEST,
    TOY_DATASET_DATES_TRAINING,
    TOY_DATASET_DATES_VALIDATION,
    TOY_DATASET_REGION,
    TOY_DATASET_VARIABLES,
    TOY_DATASET_VARIABLES_ATMOSPHERE,
    TOY_DATASET_VARIABLES_SURFACE,
)

# fmt: off
@pytest.fixture
def fake_dataset_date_ranges():
    """Fixture providing dataset date ranges for validation."""
    return {
        "toy": {
            "training": TOY_DATASET_DATES_TRAINING,
            "validation": TOY_DATASET_DATES_VALIDATION,
            "testing": TOY_DATASET_DATES_TEST,
        },
        "full": {
            "training": DATASET_DATES_TRAINING,
            "validation": DATASET_DATES_VALIDATION,
            "testing": DATASET_DATES_TEST,
        },
    }


def test_list_variables_completeness():
    """Testing that variable lists contains surface and atmospheric variables."""

    for v in DATASET_VARIABLES_SURFACE + DATASET_VARIABLES_ATMOSPHERE:
        assert v in DATASET_VARIABLES, f"ERROR - {v} is missing from DATASET_VARIABLES."

    for v in TOY_DATASET_VARIABLES_SURFACE + TOY_DATASET_VARIABLES_ATMOSPHERE:
        assert v in TOY_DATASET_VARIABLES, f"ERROR - {v} is missing from TOY_DATASET_VARIABLES."


def test_dataset_regions():
    """Testing if regions are defined correctly."""

    # Boundaries of the unprocessed Black Sea dataset
    x_max, y_max, z_max = 256, 576, 56

    for region in [TOY_DATASET_REGION, DATASET_REGION]:
        assert region["latitude"].start >= 0,      "ERROR - Latitude start is negative."
        assert region["latitude"].stop <= x_max,  f"ERROR - Latitude end is greater than maximum {x_max}."
        assert region["longitude"].start >= 0,     "ERROR - Longitude start is negative."
        assert region["longitude"].stop <= y_max, f"ERROR - Longitude end is greater than maximum {y_max}."
        assert region["level"].start >= 0,         "ERROR - Level start is negative."
        assert region["level"].stop <= z_max,     f"ERROR - Level end is greater than maximum {z_max}."


def test_dataset_date_ranges(fake_dataset_date_ranges):
    """Testing dataset date ranges for training, validation, and testing."""

    def validate_time_range(start, end, label):
        """Helper to check if a time range is valid."""
        assert (
            start < end
        ), f"ERROR - Invalid time range for {label}: start ({start}) is not before end ({end})."

    def validate_folds(dates, dataset_label):
        """Helper to ensure non-overlapping folds in dataset."""
        assert (
            dates["training"][1] <= dates["validation"][0]
        ), f"ERROR - Training dates overlap with validation dates in {dataset_label}."
        assert (
            dates["validation"][1] <= dates["testing"][0]
        ), f"ERROR - Validation dates overlap with test dates in {dataset_label}."

    # Checking everything all at once !
    for dataset_type, ranges in fake_dataset_date_ranges.items():
        for fold, (start, end) in ranges.items():
            validate_time_range(start, end, f"{dataset_type} {fold}")
        validate_folds(ranges, dataset_type)
