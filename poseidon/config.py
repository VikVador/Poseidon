r"""Configs for the project."""

from pathlib import Path

# ----- Access to raw data
#
PROJECT = Path("/gpfs/projects/acad/bsmfc/nemo4.2.0/")

PROJECT_FOLDER_DATA = PROJECT / "BSFS_BIO" / "output_HR001"
PROJECT_FILE_MASK = PROJECT / "BSFS" / "mesh_mask.nc_new59_CMCC_noAzov"

# ----- Access to personnal space
#
SCRATCH = Path("/gpfs/scratch/acad/bsmfc/victor/")
POSEIDON = SCRATCH / "poseidon"

POSEIDON_DATA = SCRATCH / "data" / "deep_learning_black_sea_3D_1995_2022.zarr"
POSEIDON_STAT = POSEIDON / "statistics" / "statistics.zarr"
POSEIDON_PTRC = POSEIDON / "paths" / "ptrc_T.txt"
POSEIDON_GRID = POSEIDON / "paths" / "grid_T.txt"
POSEIDON_MASK = POSEIDON / "mask.zarr"
POSEIDON_MESH = POSEIDON / "mesh.zarr"
