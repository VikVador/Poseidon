r"""Information about our custom datasets"""

# fmt: off
#
# ----- Global Information
#
DATASET_TRAINING   = ("1995-01", "2015-12")
DATASET_VALIDATION = ("2016-01", "2019-12")
DATASET_TEST       = ("2020-01", "2022-12")

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

# ----- Preprocessing
#
# Defining values domain for each variable
DATASET_VARIABLES_CLIPPING = {
    "DOX":      (0, None),
    "CHL":      (0, None),
    "vosaline": (0, None),
    "rho":      (0, None),
}

# ----- Datasets
#
# In case of missing values
DATASET_NAN_FILL = 0

# Black Sea
DATASET_REGION = {
    "latitude":  slice(0, 256),
    "longitude": slice(0, 576),
    "level":     slice(0, 56),
}

# Black Sea Continental Shelf (32, 128, 256)
TOY_DATASET_REGION = {
    "latitude":  slice(104, 232),
    "longitude": slice(25, 281),
    "level":     slice(0, 32),
}

# ----- Note
#
# 1. Level 32 is located at ~144.83 [m]
