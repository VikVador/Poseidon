r"""Information about our datasets"""

# fmt: off
#
# ----- Preprocessing & Datasets
#
# In case of missing values
NAN_FILL = 0

# Defining values domain for each variable
VARIABLES_CLIPPING = {
    "DOX":      (0, None),
    "CHL":      (0, None),
    "vosaline": (0, None),
    "rho":      (0, None),
}


# ----- Debug Dataset Information
#
TOY_DATASET_DATES_TRAINING   = ("2017-01-01", "2017-12-31")
TOY_DATASET_DATES_VALIDATION = ("2020-01-01", "2020-12-31")
TOY_DATASET_DATES_TEST       = ("2022-01-01", "2022-12-31")

TOY_DATASET_REGION = {
    "latitude":  slice(75, 107),
    "longitude": slice(75, 107),
    "level":     [0, 24],        # 0.25 [m] / 46 [m]
}

TOY_DATASET_VARIABLES_ATMOSPHERE = [
    "DOX",
]

TOY_DATASET_VARIABLES_SURFACE = [
    # None
]

TOY_DATASET_VARIABLES = \
    TOY_DATASET_VARIABLES_ATMOSPHERE + TOY_DATASET_VARIABLES_SURFACE


# ----- Black Sea Continental Shelf Dataset Information
#
DATASET_DATES_TRAINING       = ("1995-01-01", "2017-12-31")
DATASET_DATES_VALIDATION     = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST           = ("2021-01-01", "2022-12-31")

DATASET_REGION = {
    "latitude":  slice(104, 232),
    "longitude": slice(25, 281),
    "level":     slice(0, 56),
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
