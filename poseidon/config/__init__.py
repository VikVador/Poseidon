r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
# ----- Simulation
#
SIMULATION      = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")
SIMULATION_DATA = SIMULATION / "BSFS_BIO" / "output_HR001"
SIMULATION_MASK = SIMULATION / "BSFS"     / "mesh_mask.nc_new59_CMCC_noAzov"

# ----- Poseidon
#
SCRATCH    = Path("/gpfs/scratch/acad/bsmfc/victor/")
POSEIDON   = SCRATCH  / "poseidon"

PATH_DATA  = SCRATCH  / "data"       / "deep_learning_black_sea_3D_1995_2022.zarr"
PATH_OBS   = SCRATCH  / "data"       / "observations"
PATH_STAT  = POSEIDON / "statistics" / "statistics.zarr"
PATH_PTRC  = POSEIDON / "paths"      / "ptrc_T.txt"
PATH_GRID  = POSEIDON / "paths"      / "grid_T.txt"
PATH_MASK  = POSEIDON / "mask.zarr"
PATH_MESH  = POSEIDON / "mesh.zarr"
PATH_MODEL = POSEIDON / "models"

# ----- Observations
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
