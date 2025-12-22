r"""Script to download a dataset from the Copernicus Marine Service."""

import argparse
import copernicusmarine
import xarray as xr

from dawgz import job, schedule
from functools import partial
from pathlib import Path

# isort: split
from poseidon.config import PATH_MASK_B

YEARS = {
    "chl": (1998, 2024),
    "sss": (2011, 2024),
    "sst": (2008, 2024),
    "ssh": (2002, 2004),
}

VARIABLES = {
    "chl": {
        "L3": ["CHL"],
        "L4": ["CHL"],
    },
    "sss": {
        "L3": ["Sea_Surface_Salinity", "Sea_Surface_Salinity_Error"],
        "L4": ["sos", "sos_error"],
    },
    "sst": {
        "L3": ["adjusted_sea_surface_temperature", "adjusted_standard_deviation_error"],
        "L4": ["analysed_sst"],
    },
    "ssh": {
        "L3": ["mdt"],
        "L4": ["adt"],
    },
}

DATASET_IDS = {
    "chl": {
        "L3": "cmems_obs-oc_blk_bgc-plankton_my_l3-multi-1km_P1D",
        "L4": "cmems_obs-oc_blk_bgc-plankton_my_l4-gapfree-multi-1km_P1D",
    },
    "sss": {
        "L3": "cmems_obs-mob_glo_phy-sss_mynrt_smos-asc_P1D",
        "L4": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
    },
    "sst": {
        "L3": "SST_BS_SST_L3S_NRT_OBSERVATIONS_010_013_a",
        "L4": "cmems_SST_BS_SST_L4_REP_OBSERVATIONS_010_022",
    },
    "ssh": {
        "L3": "cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S",
        "L4": "cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D",
    },
}

# Coordinates for spatial filtering
mask = xr.open_zarr(PATH_MASK_B).load()

BS_LAT_MIN, BS_LAT_MAX, BS_SHELF_LAT_MIN, BS_SHELF_LAT_MAX = (
    mask.latitude.values[0],
    mask.latitude.values[256],
    mask.latitude.values[104],
    mask.latitude.values[232],
)

BS_LON_MIN, BS_LON_MAX, BS_SHELF_LON_MIN, BS_SHELF_LON_MAX = (
    mask.longitude.values[0],
    mask.longitude.values[576],
    mask.longitude.values[25],
    mask.longitude.values[281],
)

REGION_BOUNDS = {
    "shelf": (BS_SHELF_LON_MIN, BS_SHELF_LON_MAX, BS_SHELF_LAT_MIN, BS_SHELF_LAT_MAX),
    "black_sea": (BS_LON_MIN, BS_LON_MAX, BS_LAT_MIN, BS_LAT_MAX),
}


def download_dataset(path_output: Path, variable: str, product: str, region: str) -> None:
    r"""Downloads datasets of satellite data from the Copernicus Marine Service.

    Platform:
    | https://data.marine.copernicus.eu/products?facets=areas%7EBlack+Sea

    Arguments:
        path_output: Path of folder to save the output datasets.
        variable: Variable to download (e.g., 'sst', 'ssh', 'chl').
        product: Product type ('L3' or 'L4').
        region: Region focus ('shelf' or 'black_sea').
    """

    y_start, y_end = YEARS[variable]
    year_start = [str(i) for i in range(y_start, y_end - 1)]
    year_end = [str(i) for i in range(y_start + 1, y_end)]
    min_lon, max_lon, min_lat, max_lat = REGION_BOUNDS[region]

    for ys, ye in zip(year_start, year_end):
        ds = f"{ys}-01-01T00:00:00"
        de = f"{ye}-12-31T00:00:00"
        output_file = path_output / f"CMEMS_{variable.upper()}_{product}_{region.upper()}_{ys}_{ye}.nc"

        copernicusmarine.subset(
            dataset_id=DATASET_IDS[variable][product],
            variables=VARIABLES[variable][product],
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            start_datetime=ds,
            end_datetime=de,
            output_filename=str(output_file),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloads datasets of satellite data from the Copernicus Marine Service."
    )

    parser.add_argument(
        "--path_output",
        "-o",
        type=Path,
        required=True,
        help="Path of folder in which save the output datasets.",
    )
    parser.add_argument(
        "--variable",
        "-v",
        type=str,
        required=True,
        choices=["sst", "sss", "ssh", "chl"],
        help="Variable to download ('sst', 'sss', 'ssh', 'chl').",
    )
    parser.add_argument(
        "--product",
        "-p",
        type=str,
        required=True,
        choices=["L3", "L4"],
        help="Product type ('L3' or 'L4').",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        required=True,
        choices=["shelf", "black_sea"],
        help="Region focus ('shelf' or 'black_sea').",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="async",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args = parser.parse_args()
    args.path_output.mkdir(parents=True, exist_ok=True)

    dawgz_download = partial(
        download_dataset,
        args.path_output,
        args.variable,
        args.product,
        args.region,
    )

    schedule(
        job(
            dawgz_download,
            cpus=2,
            mem="16GB",
            name="POSEIDON-DOWNLOAD",
            time="00-01:00:00",
            account="bsmfc",
            partition="batch",
        ),
        name="POSEIDON-DOWNLOAD",
        export="ALL",
        backend=args.backend,
    )
