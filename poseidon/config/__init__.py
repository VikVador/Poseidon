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
PATH_STAT  = POSEIDON / "statistics" / "statistics.zarr"
PATH_PTRC  = POSEIDON / "paths"      / "ptrc_T.txt"
PATH_GRID  = POSEIDON / "paths"      / "grid_T.txt"
PATH_MASK  = POSEIDON / "mask.zarr"
PATH_MESH  = POSEIDON / "mesh.zarr"
PATH_MODEL = POSEIDON / "models"
