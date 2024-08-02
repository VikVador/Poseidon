# -------------------------------------------------------------
#
#        |
#       / \                 ESA - MITHO PROJECT
#      / _ \
#     |.o '.|      "GENERATIVE MODELS FOR HYPOXIA FORECASTING"
#     |'._.'|
#     |     |               by VICTOR MANGELEER
#   ,'|  |  |`.
#  /  |  |  |  \                2023-2024
#  |,-'--|--'-.|
#
# --------------------------------------------------------------
# https://eo4society.esa.int/projects/mitho/
#
# Libraries
import ast
import wandb
import argparse
import numpy as np
import xarray as xr
from dawgz import Job, schedule

# --------------------------------
#        Script Parameters
# --------------------------------
#
# Wheter or not doing
USE_WANDB         = 1
USE_CUSTOM_REGION = 1
SAVE_MASK         = 1

# Time Period
YEAR_START = 1982
YEAR_END   = 1982

# Region
X_START, X_END = 108, 236
Y_START, Y_END = 108, 236
Z_END          = 32

# Name of the final destination folder
FOLDER_NAME = "training"

# High Resolution Data to load
DATA_PATH = "~/../../../../projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/"

# Main folder containing the different datasets
SAVE_PATH = "~/../../../../scratch/acad/bsmfc/victor/data/deep_learning_3D/"

# Path to the mask
MASK_PATH = "~/../../../../projects/acad/bsmfc/nemo4.2.0/BSFS/mesh_mask.nc_new59_CMCC_noAzov"

# Variables (Can be either surface of volumetric)
DATA_VARIABLES = [
    "votemper",
    "vosaline",
    "CHL",
    "DOX",
    "ssh",
    ]

# Variables to clip (to avoid negative values if not physical)
DATA_CLIPPING = [
    "DOX",
    ]

# --------------------------------
#            Function
# --------------------------------
#
def preprocessing(useWandb: bool,
                  useCustomRegion: bool,
                  saveMask: bool,
                  foldername: str,
                  year_start: int,
                  year_end: int,
                  x_start: int,
                  x_end: int,
                  y_start: int,
                  y_end: int,
                  z_end: int,
                  config: dict = None):
    """
    Preprocesses the data, Black Sea simulation of
    its physics and biogeochemistry, in 3D and save
    everything as .netcdf4 files.

    Args:
        useWandb:        Wheter or not using Weights and Biases.
        useCustomRegion: Wheter or not using a custom region.
        saveMask:        Wheter or not saving the mask.
        foldername:      Name of the final destination folder.
        year_start:      Starting year of the time period.
        year_end:        Ending year of the time period.
        x_start:         Starting index of the x-axis.
        x_end:           Ending index of the x-axis.
        y_start:         Starting index of the y-axis.
        y_end:           Ending index of the y-axis.
        z_end:           Ending index of the z-axis.
        config:          Configuration of the script to send to WandB.

    Returns:
        None

    """

    # --------------------------------
    #            Security
    # --------------------------------
    #
    # 1. Time
    assert 1982 <= year_start <= 2022, "Time period must be between 1982 and 2023 (YEAR_START)"
    assert 1982 <= year_end   <= 2022, "Time period must be between 1982 and 2023 (YEAR_END)"
    assert year_start <= year_end,     "Ending time period must be equal or greater than starting (YEAR_START <= YEAR_END)"

    # 2. Region indices
    assert 0 <= x_start < X_END <= 576, "X_START and X_END must be between 0 and 576"
    assert 0 <= Y_START < Y_END <= 256, "Y_START and Y_END must be between 0 and 256"
    assert 0 <= z_end <= 59,            "Z_END must be between 0 and 59"

    # 3. Folders
    assert foldername in ["training", "validation", "test"], "Folder name must be either 'training', 'validation' or 'testing'"

    # --------------------------------
    #    Preprocessing (1) - MASK
    # --------------------------------
    # Extracting mapping between dates and files
    with open('paths/grid_T.txt', 'r') as file:
        data_physics = ast.literal_eval(file.read())

    with open('paths/ptrc_T.txt', 'r') as file:
        data_biogeochemistry = ast.literal_eval(file.read())

    # ----
    # MASK
    # ----
    # Loading the dataset storing the mask
    dataset_mask_complete = xr.open_dataset(MASK_PATH, engine= "netcdf4")

    # Extracting the mask (indices define the bottom)
    mask = dataset_mask_complete["mbathy"][0].values

    # Extracting the region of interest
    mask = mask[y_start:y_end, x_start:x_end] if useCustomRegion else mask

    # Conversion to a 3D mask with specific depth
    mask_matrix = np.ones((z_end, mask.shape[0], mask.shape[1])) if useCustomRegion else np.ones((59, mask.shape[0], mask.shape[1]))

    # Filling the 3D mask (1 = Black Sea, 0 = Land)
    for l in range(mask_matrix.shape[0]):
        mask_matrix[l, mask <= l] = 0

    # --------------------------------
    #    Preprocessing (2) - Data
    # --------------------------------
    # Initialization of Weights and Biases
    wandb.init(project = 'Poseidon - Preprocessing', config= config, mode= 'disabled' if not useWandb else 'online')

    # Used to store statistics
    means = [0 for i in range(len(DATA_VARIABLES))]
    vars  = [0 for i in range(len(DATA_VARIABLES))]

    # Counts the number of samples (used for temporal average)
    n_samples = 0

    # Computing statistics, standardizing the data and saving it
    for stat_index, stat_name in enumerate(["Mean", "Variance", "Standardization"]):

        # Computing for each month of the year
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):

                # Sending information to Weights and Biases
                wandb.log({f"{stat_name}/year": year, f"{stat_name}/month": month})

                # Creating the key to access the data
                key = f"{year:04d}-{month:02d}"

                # Creating the paths
                paths = [DATA_PATH + p for p in data_physics[key] + data_biogeochemistry[key]]

                # Loading view of the data
                data = xr.open_mfdataset(paths, combine = 'by_coords')

                # Extracting the variables
                data = data[DATA_VARIABLES]

                # Updating the number of samples (used for temporal average)
                n_samples += data.sizes['time_counter'] if stat_index == 0 else 0

                # Used to store data and its name (Only for standardization)
                preprocessed_data = []

                # ------- Looping over the variables --------
                for i, variable in enumerate(data.data_vars):

                    # Extracting the data and clipping it
                    var_data = data[variable].values if variable not in DATA_CLIPPING else np.clip(data[variable].values, 0, np.inf)

                    # Total number of cells
                    n_points = np.sum(mask_matrix) if var_data.ndim == 4 else np.sum(mask_matrix[0])

                    # Conversion of surface variables to volume
                    if var_data.ndim == 3:
                        t, y, x = var_data.shape
                        domain = np.zeros((t, 59, y, x))
                        domain[:, 0, ...] = var_data
                        var_data = domain

                    # Extracting the region of interest
                    var_data = var_data[:, :z_end, y_start:y_end, x_start:x_end] if useCustomRegion else var_data

                    # Setting eveything in the land to null value
                    var_data[:, mask_matrix == 0] = 0

                    # --------- Statistics (1) ---------
                    if stat_index == 0:

                        # Computing the spatial mean
                        means[i] += np.nansum(var_data)/n_points

                    elif stat_index == 1:

                        # Computing the variance pixel-wise
                        var_data = np.power(var_data - means[i], 2)

                        # Removing biased values (land is now not set to 0)
                        var_data[:, mask_matrix == 0] = 0

                        # Computing the variance
                        vars[i] += np.nansum(var_data)/n_points

                    elif stat_index == 2:

                        # Standardizing the data
                        var_data = (var_data - means[i]) / stds[i]

                        # Removing biased values (land is now not set to 0)
                        var_data[:, mask_matrix == 0] = 0

                        # Flipping the data to display Black Sea correctly
                        var_data = np.flip(var_data, axis = 2)

                        # Adding the data to the list
                        preprocessed_data.append([variable, var_data])

                    else:
                        print("ERROR (generating.py) - Unknown Step")

                # --------- Saving ---------
                #
                if stat_index == 2:

                    # Creating the final dataset containing the preprocessed data and associated dates
                    dataset_final = xr.Dataset(
                        {
                            preprocessed_data[i][0]: (["time_counter", "nav_lev", "y", "x"], preprocessed_data[i][1]) for i in range(len(preprocessed_data))
                        },
                        coords = {
                            "time_counter": data["time_counter"],
                            "nav_lev": np.arange(0, preprocessed_data[0][1].shape[1]),
                            "y": np.arange(0, preprocessed_data[0][1].shape[2]),
                            "x": np.arange(0, preprocessed_data[0][1].shape[3])
                        }
                    )

                    # Saving the final dataset
                    dataset_final.to_netcdf(SAVE_PATH + foldername + f"/deep_learning_black_sea_data_3D_{key}.nc4", mode= "w", engine = "netcdf4")

        # --------- Stastistics (2) ---------
        #
        if stat_index == 0:

            # Computing the spatial-temporal mean
            means = [mean / n_samples for mean in means]

        if stat_index == 1:

            # Computing the spatial-temporal variance
            vars = [var / n_samples for var in vars]

            # Computing the spatial-temporal standard deviation
            stds = [np.sqrt(var) for var in vars]

    # --------------------------------
    #    Preprocessing (3) - Others
    # --------------------------------
    #
    # Flipping the data to display Black Sea correctly
    mask_matrix = np.flip(mask_matrix, axis = 1)

    # Creating a meshgrid
    x_mesh, y_mesh, z_mesh = np.meshgrid(np.arange(0, mask_matrix.shape[2]), np.arange(0, mask_matrix.shape[1]), np.arange(0, mask_matrix.shape[0]))

    # Depth is first, than y and x
    x_mesh, y_mesh, z_mesh = np.swapaxes(x_mesh, 0, 2), np.swapaxes(y_mesh, 0, 2), np.swapaxes(z_mesh, 0, 2)
    x_mesh, y_mesh, z_mesh = np.swapaxes(x_mesh, 1, 2), np.swapaxes(y_mesh, 1, 2), np.swapaxes(z_mesh, 1, 2)

    # Creating the final dataset containing the mask and mesh
    dataset_utils= xr.Dataset(
        {
            "mask"  : (["nav_lev", "y", "x"], mask_matrix),
            "x_mesh": (["nav_lev", "y", "x"], x_mesh),
            "y_mesh": (["nav_lev", "y", "x"], y_mesh),
            "z_mesh": (["nav_lev", "y", "x"], z_mesh)
        },
        coords = {
            "x"      : np.arange(0, mask_matrix.shape[2]),
            "y"      : np.arange(0, mask_matrix.shape[1]),
            "nav_lev": np.arange(0, mask_matrix.shape[0])
        }
    )

    # Saving the mask
    dataset_utils.to_netcdf(SAVE_PATH + f"deep_learning_black_sea_data_3D_utils.nc4", mode= "w", engine= "netcdf4") if saveMask else None


if __name__ == '__main__':

    # Parsing the arguments
    parser = argparse.ArgumentParser(description="Preprocessing of the data for the Deep Learning model (3D)")

    parser.add_argument("--dawgz",            type=int, default=1,                 help="Wheter or not running it as a Dawgz job.")
    parser.add_argument("--foldername",       type=str, default=FOLDER_NAME,       help="Name of the final destination folder.")
    parser.add_argument("--useWandb",         type=int, default=USE_WANDB,         help="Wheter or not using Weights and Biases.")
    parser.add_argument("--useCustomRegion",  type=int, default=USE_CUSTOM_REGION, help="Wheter or not using a custom region.")
    parser.add_argument("--saveMask",         type=int, default=SAVE_MASK,         help="Wheter or not saving the mask.")
    parser.add_argument("--year_start",       type=int, default=YEAR_START,        help="Starting year of the time period.")
    parser.add_argument("--year_end",         type=int, default=YEAR_END,          help="Ending year of the time period.")
    parser.add_argument("--x_start",          type=int, default=X_START,           help="Starting index of the x-axis.")
    parser.add_argument("--x_end",            type=int, default=X_END,             help="Ending index of the x-axis.")
    parser.add_argument("--y_start",          type=int, default=Y_START,           help="Starting index of the y-axis.")
    parser.add_argument("--y_end",            type=int, default=Y_END,             help="Ending index of the y-axis.")
    parser.add_argument("--z_end",            type=int, default=Z_END,             help="Ending index of the z-axis.")

    args = parser.parse_args()

    # Dawgz Preprocessing Function Tool
    def dawgz_preprocessing():
        preprocessing(args.useWandb, args.useCustomRegion, args.saveMask, args.foldername, args.year_start, args.year_end, args.x_start, args.x_end, args.y_start, args.y_end, args.z_end, vars(args))

    # Launching Dawgz
    schedule(
            Job(
                dawgz_preprocessing,
                cpus=4,
                gpus=1,
                mem='120GB',
                name='PREPROCESSING',
                time='00-12:00:00',
                account='bsmfc',
                partition='gpu',
            ),
            name='PREPROCESSING',
            export='ALL',
            shell="/bin/sh",
            backend='slurm' if args.dawgz else 'async',
        )