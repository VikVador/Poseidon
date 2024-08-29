r"""Configs for the project."""

from pathlib import Path

SCRATCH = Path("/gpfs/scratch/acad/sail")
PROJECT = SCRATCH / "aang"

DATASET_ERA5 = SCRATCH / "data" / "era5_1999-2019-1h-1440x721-gencast.zarr"
DATASET_STAT = PROJECT / "data" / "fake_stats.zarr"  # TO BE CHANGED
