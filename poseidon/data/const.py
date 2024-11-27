r"""Information about our datasets"""

# fmt: off
#
# ----- Global Information
#
TOY_DATASET_DATES_TRAINING   = ("2014-01-01", "2015-12-31")
TOY_DATASET_DATES_VALIDATION = ("2019-01-01", "2019-12-31")
TOY_DATASET_DATES_TEST       = ("2022-01-01", "2022-12-31")

DATASET_DATES_TRAINING       = ("1995-01-01", "2015-12-31")
DATASET_DATES_VALIDATION     = ("2016-01-01", "2019-12-31")
DATASET_DATES_TEST           = ("2020-01-01", "2022-12-31")

DATASET_VARIABLES_ATMOSPHERE = [
    "DOX",
    "CHL",
    "vosaline",
    "rho",
    "votemper",
]

DATASET_VARIABLES_SURFACE = [
    "ssh",
]

DATASET_VARIABLES = \
    DATASET_VARIABLES_ATMOSPHERE + DATASET_VARIABLES_SURFACE

TOY_DATASET_REGION = {
    "latitude":  slice(104, 232),
    "longitude": slice(25, 281),
    "level":     slice(0, 32), # ~144.83 [m]
}

DATASET_REGION = {
    "latitude":  slice(0, 256),
    "longitude": slice(0, 576),
    "level":     slice(0, 56),
}

# ----- Preprocessing & Datasets
#
# In case of missing values
DATASET_NAN_FILL = 0

# Defining values domain for each variable
DATASET_VARIABLES_CLIPPING = {
    "DOX":      (0, None),
    "CHL":      (0, None),
    "vosaline": (0, None),
    "rho":      (0, None),
}
