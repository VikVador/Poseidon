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
)


@pytest.fixture
def dataset_date_ranges():
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


def test_dataset_variables():
    """Testing dataset variable definitions."""
    for var in DATASET_VARIABLES_ATMOSPHERE:
        assert var in DATASET_VARIABLES, f"ERROR - {var} missing from DATASET_VARIABLES."
    for var in DATASET_VARIABLES_SURFACE:
        assert var in DATASET_VARIABLES, f"ERROR - {var} missing from DATASET_VARIABLES."


def test_dataset_regions():
    """testing region definitions for toy and full datasets."""
    for key in TOY_DATASET_REGION:
        assert (
            TOY_DATASET_REGION[key].start >= DATASET_REGION[key].start
        ), f"ERROR - Toy dataset region {key} starts outside full dataset region."
        assert (
            TOY_DATASET_REGION[key].stop <= DATASET_REGION[key].stop
        ), f"ERROR - Toy dataset region {key} ends outside full dataset region."


def test_dataset_date_ranges(dataset_date_ranges):
    """testing dataset date ranges for training, validation, and testing."""

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

    for dataset_type, ranges in dataset_date_ranges.items():
        for fold, (start, end) in ranges.items():
            validate_time_range(start, end, f"{dataset_type} {fold}")
        validate_folds(ranges, dataset_type)
