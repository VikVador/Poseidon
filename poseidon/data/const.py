r"""Information about our datasets"""

# fmt: off
#
# ----- Preprocessing & Datasets
#
LAND_VALUE = 0

# Defining each variable physical domain
VARIABLES_CLIPPING = {
    "DOX":      (0, None),
    "CHL":      (0, None),
    "vosaline": (0, None),
}


# ----- Dataset: Black Sea Continental Shelf
#
TOY_DATASET_DATES_TRAINING   = ("2017-01-01", "2017-12-31")
TOY_DATASET_DATES_VALIDATION = ("2018-01-01", "2020-12-31")
TOY_DATASET_DATES_TEST       = ("2021-01-01", "2022-12-31")

TOY_DATASET_REGION = {
    "latitude":  slice(104, 232),
    "longitude": slice(25, 281),
    "level": slice(0, 32),
}

TOY_DATASET_VARIABLES_ATMOSPHERE = [
    "DOX",
    "CHL",
    "vosaline",
    "votemper",
]

TOY_DATASET_VARIABLES_SURFACE = [
    "ssh",
]

TOY_DATASET_VARIABLES = \
    TOY_DATASET_VARIABLES_ATMOSPHERE + TOY_DATASET_VARIABLES_SURFACE


# ----- Dataset: Black Sea
#
DATASET_DATES_TRAINING       = ("2017-01-01", "2017-12-31")
DATASET_DATES_VALIDATION     = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST           = ("2021-01-01", "2022-12-31")

DATASET_REGION = {
    "latitude":  slice(1, 257),
    "longitude": slice(1, 577),
    "level": slice(0, 32),
}

DATASET_VARIABLES_ATMOSPHERE = [
    "DOX",
    "CHL",
    "vosaline",
    "votemper",
]

DATASET_VARIABLES_SURFACE = [
    "ssh",
]

DATASET_VARIABLES = \
    DATASET_VARIABLES_ATMOSPHERE + DATASET_VARIABLES_SURFACE
