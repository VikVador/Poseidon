r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
# ----- Simulation
#
SIMULATION      = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")
SIMULATION_DATA = SIMULATION / "BSFS_BIO" / "output_HR001"
SIMULATION_MASK = SIMULATION / "BSFS"     / "mesh_mask.nc_new59_CMCC_noAzov"

# ----- Main Folders
#
# Personnal
PATH_MAIN_LOCAL = Path("/gpfs/home/acad/ulg-mast/vmangele/")

# MAST-DB (non-wiping)
PATH_MAIN_PROJECT = Path("/gpfs/projects/acad/bsmfc/Obs/mastdb/vmangele/")

# Scratch (wiping)
PATH_MAIN_SCRATCH = Path("/gpfs/scratch/acad/bsmfc/vmangele/")

# ======================================
#           P O S E I D O N
# ======================================
#
# ----- Main Folders
#
PATH_POS_LOCAL   = PATH_MAIN_LOCAL   / "poseidon"
PATH_POS_PROJECT = PATH_MAIN_PROJECT / "poseidon"
PATH_POS_SCRATCH = PATH_MAIN_SCRATCH / "poseidon"

# ----- Experiments
#
PATH_EXP       = PATH_POS_PROJECT / "experiments"
PATH_EXP_OBS   = PATH_EXP / "observations"
PATH_EXP_MASKS = PATH_EXP / "masks"

# ----- Others
#
PATH_MODEL  = PATH_POS_SCRATCH / "models"
PATH_PTRC   = PATH_POS_PROJECT / "paths"    / "ptrc_T.txt"
PATH_GRID   = PATH_POS_PROJECT / "paths"    / "grid_T.txt"
PATH_OBS    = PATH_POS_PROJECT / "datasets" / "observations"
PATH_MESH   = PATH_POS_PROJECT / "datasets" / "structure" / "mesh_black_sea.zarr"
PATH_MASK_B = PATH_POS_PROJECT / "datasets" / "structure" / "mask_black_sea.zarr"
PATH_MASK_V = PATH_POS_PROJECT / "datasets" / "structure" / "mask_variables.zarr"
PATH_STAT   = PATH_POS_PROJECT / "datasets" / "statistics" / "statistics_black_sea_HR001_1980_2017.zarr"

# ----- Dataset | Deep Learning
#
PATH_DATA = PATH_POS_SCRATCH / "datasets" / "deep_learning_black_sea_HR001_1980_2023.zarr"

# ----- Dataset | Real Observations
#
PATH_OBSERVATIONS_FLOATS = {
    "shelf": {
        "oxygen":      PATH_OBS / "observations_1980_2025_floats_oxygen_shelf.zarr",
        "salinity":    PATH_OBS / "observations_1980_2025_floats_salinity_shelf.zarr",
        "temperature": PATH_OBS / "observations_1980_2025_floats_temperature_shelf.zarr",
    },
    "black_sea": {
        "oxygen":      PATH_OBS / "observations_1980_2025_floats_oxygen.zarr",
        "salinity":    PATH_OBS / "observations_1980_2025_floats_salinity.zarr",
        "temperature": PATH_OBS / "observations_1980_2025_floats_temperature.zarr",
    },
}

PATH_OBSERVATIONS_SATELLITE = {
    "shelf": {
        "chlorophyll": {
            "L3": PATH_OBS / "observations_1998_2022_satellite_chlorophyll_L3_shelf.zarr",
            "L4": PATH_OBS / "observations_1998_2022_satellite_chlorophyll_L4_shelf.zarr",
        },
        "salinity": {
            "L3": PATH_OBS / "observations_2011_2020_satellite_salinity_L3_shelf.zarr",
            "L4": PATH_OBS / "observations_2011_2019_satellite_salinity_L4_shelf.zarr",
        },
        "sea_surface_height": {
            "L3": None,
            "L4": PATH_OBS / "observations_1998_2022_satellite_sea_surface_height_L4_shelf.zarr",
        },
        "temperature": {
            "L3": PATH_OBS / "observations_1982_2022_satellite_temperature_L3_shelf.zarr",
            "L4": PATH_OBS / "observations_1982_2022_satellite_temperature_L4_shelf.zarr",
        },
    },
    "black_sea": {
        "chlorophyll": {
            "L3": PATH_OBS / "observations_1998_2022_satellite_chlorophyll_L3.zarr",
            "L4": PATH_OBS / "observations_1998_2022_satellite_chlorophyll_L4.zarr",
        },
        "salinity": {
            "L3": PATH_OBS / "observations_2011_2020_satellite_salinity_L3.zarr",
            "L4": PATH_OBS / "observations_2011_2019_satellite_salinity_L4.zarr",
        },
        "sea_surface_height": {
            "L3": None,
            "L4": PATH_OBS / "observations_1998_2022_satellite_sea_surface_height_L4.zarr",
        },
        "temperature": {
            "L3": PATH_OBS / "observations_1982_2022_satellite_temperature_L3.zarr",
            "L4": PATH_OBS / "observations_1982_2022_satellite_temperature_L4.zarr",
        },
    },
}
