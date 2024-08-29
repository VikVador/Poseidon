r"""A collection of tools for the project."""

import numpy as np
import pandas as pd
import re
import torch


def assert_date_format(date_string: str) -> None:
    r"""Asserts that the date string is in the correct format (YYYY-MM-DD).

    Arguments:
        date_string (str): Date string to check.

    Raises:
        ValueError: If the date string does not match the pattern
    """

    # YYYY-MM-DD
    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"

    if not re.match(pattern, date_string):
        raise ValueError("The format is incorrect. The date should be in YYYY-MM-DD format.")


def get_date_features(date: np.datetime64) -> torch.Tensor:
    r"""Extracts year, month, day, and hour from a numpy.datetime64 object.

    Arguments:
        date (np.datetime64): A numpy.datetime64 object representing a specific date and time.

    Returns:
        torch.Tensor: A tensor containing the year, month, day, and hour extracted from the input date.
    """
    timestamp = pd.to_datetime(date)
    return torch.as_tensor([timestamp.year, timestamp.month, timestamp.day, timestamp.hour])
