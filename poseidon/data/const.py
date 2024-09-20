r"""Custom Black Sea dataset information (1980 to 2022)."""

# -- Information --
#
DATASET_TRAINING_DATE_START = "1995-01"
DATASET_TRAINING_DATE_END = "2015-12"

DATASET_VALIDATION_DATE_START = "2016-01"
DATASET_VALIDATION_DATE_END = "2019-12"

DATASET_TESTING_DATE_START = "2020-01"
DATASET_TESTING_DATE_END = "2022-12"

DATASET_VARIABLES = [
    "DOX",
    "CHL",
    "vosaline",
    "rho",
    "ssh",
    "votemper",
]

# -- Preprocessing --
#
DATASET_VARIABLES_CLIPPING = {
    "DOX": (0, None),
    "CHL": (0, None),
    "vosaline": (0, None),
    "rho": (0, None),
}

DATASET_VARIABLES_SURFACE = [
    "ssh",
]
