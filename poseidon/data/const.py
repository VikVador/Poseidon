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
DATASET_DATES_TRAINING       = ("1998-01-01", "2017-12-31")
DATASET_DATES_VALIDATION     = ("2018-01-01", "2020-12-31")
DATASET_DATES_TEST           = ("2021-01-01", "2022-12-31")

DATASET_REGION = {
    "latitude":  slice(104, 232), # 43.500 to 46.675  [°N]
    "longitude": slice(25,  281), # 28.025 to 34.400  [°E]
    "level":     slice(0,    32), # 0.255  to 125.233 [m]
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

# ----- Observations
#
OBSERVATIONS_RESOLUTION = {
    "chl":      (128, 256), # 0.025  [°] × 0.025  [°] ~ 2.5 [km] × 2.5 [km]
    "vosaline": ( 13,  26), # 0.25   [°] × 0.25   [°] ~ 25  [km] × 25  [km]
    "votemper": ( 63, 129), # 0.05   [°] × 0.05   [°] ~ 5   [km] × 5   [km]
    "ssh":      ( 51, 102), # 0.0625 [°] × 0.0625 [°] ~ 7   [km] × 7   [km]
}
