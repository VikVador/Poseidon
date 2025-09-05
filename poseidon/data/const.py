r"""Information about our dataset"""

# fmt: off
#
# ----- Preprocessing
#
# Replacing land values (NaNs) by it
LAND_VALUE = 0

# Defining each variable physical domain (it corrects emulation errors)
VARIABLES_CLIPPING = {
    "DOX":      (0, None),
    "CHL":      (0, None),
    "vosaline": (0, None),
}

# ----- Dataset: Black Sea Continental Shelf
#
DATASET_DATES_TRAINING       = ("1995-01-01", "2017-12-31")
DATASET_DATES_VALIDATION     = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST           = ("2021-01-01", "2022-12-31")

DATASET_REGION = {
    "latitude":  slice(104, 232),
    "longitude": slice(25, 281),
    "level": slice(0, 32),
}

DATASET_VARIABLES_OCEAN = [
    "DOX",
    "CHL",
    "vosaline",
    "votemper",
]

DATASET_VARIABLES_SURFACE = [
    "ssh",
]

DATASET_VARIABLES = \
    DATASET_VARIABLES_OCEAN + DATASET_VARIABLES_SURFACE
