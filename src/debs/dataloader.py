# -------------------------------------------------------
#
#        |
#       / \
#      / _ \                  ESA - PROJECT
#     |.o '.|
#     |'._.'|          BLACK SEA DEOXYGENATION EMULATOR
#     |     |
#   ,'|  |  |`.             BY VICTOR MANGELEER
#  /  |  |  |  \
#  |,-'--|--'-.|                2023-2024
#
#
# -------------------------------------------------------
#
# Documentation
# -------------
# A tool to create a dataloader for Black Sea dataset, i.e. used for training, validating and testing the model
#
import numpy as np

# Torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def preprocess(data: np.array, mask: np.array):
    r"""Used to mask the NaNs, mask the land and normalize the data"""

    # Rescale the data to ensure non-negative values
    data += np.abs(np.nanmin(data)) if np.nanmin(data) < 0 else 0

    # Set the land values to 0
    data[:, mask == 0] = 0

    # Replace NaNs with 0
    data[np.isnan(data)] = 0

    # Normalize the data
    normalized_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    return normalized_data

def merge_timeseries_and_patches(data: np.array):
    r"""Used to merge the timeseries and patches dimensions"""

    # Swaping axes (needed to concatenate along patch dimensino during reshaping), i.e. (t, v, p, r, r) to (t, p, v, r, r)
    data = np.swapaxes(data, 1, 2)

    # Retrieving all the dimensions for reshaping (easier to understand what is happening)
    t, p, v, res_x, res_y = data.shape

    # Merging timeseries and patches dimensions
    return data.reshape(t * p, v, res_x, res_y)

class BlackSea_Dataloader():
    r"""A dataloader for Black Sea dataset, i.e. used for training, validating and testing the model."""

    def __init__(self, x: list,
                       y: np.array,
                    mask: np.array,
                    mode: str  = "spatial",
              resolution: int  = 64,
                  window: int  = 1,
              window_oxy: int  = 7,
           datasets_size: list = [0.5, 0.25],
                    seed: int  = 69):

        # ------------------------------------------------
        #                  PREPROCESSING (1)
        # ------------------------------------------------
        #
        # Concatenate inputs and output
        x = np.stack([y] + x, axis = 1)

        # Removing dimensions to be a power of 2
        x    = x[:, :, :-2, :-2]
        mask = mask[:-2, :-2]

        # Current shape of the input (1)
        t, v, x_res, y_res = x.shape

        # Security
        assert mode in ["spatial", "temporal"], f"ERROR (BlackSea_Dataloader) Mode must be either 'spatial', 'temporal' ({mode})"
        assert resolution % 2 == 0,             f"ERROR (BlackSea_Dataloader) Resolution must be a multiple of 2 ({resolution})"
        assert resolution <= x_res/2,           f"ERROR (BlackSea_Dataloader) Resolution must be smaller than half the input resolution ({resolution} < {x_res/2})"
        assert window <= int(t/3 - 1),          f"ERROR (BlackSea_Dataloader) Window must be smaller than a third of the input time scale ({window} < {int(t/3 - 1)})"

        # Preprocessing the data
        for i in range(v):
            x[:, i, :, :] = preprocess(x[:, i, :, :], mask)

        # Number of patches along the x-, y- dimensions and total number of possible patches
        nb_patches_x, nb_patches_y, total_patches = int(x_res/resolution), int(y_res/resolution), int(x_res/resolution) * int(y_res/resolution)

        # Extracting patches from the input of of a given resolution
        x = [x[:, :, i * resolution : (i + 1) * resolution, j * resolution : (j + 1) * resolution] for i in range(nb_patches_x) for j in range(nb_patches_y)]

        # Concatenation of the inputs (t, variable, x, y) into (t, variables, number of patches, resolution, resolution)
        x = np.stack(x, axis = 2)

        # Separation of the x and y data (to avoid a mess with timeseries)
        y = x[:, 0,  :, :, :]
        x = x[:, 1:, :, :, :]

        # Input - Creation of the input time series, i.e. (index, variable(s)_{t, t + window}, number of patches, resolution, resolution)
        x = np.stack([x[i : i + window, :, :, :, :] for i in range(t - window - window_oxy)], axis = 0).reshape(t - window - window_oxy, (v - 1) * window, total_patches, resolution, resolution)

        # Output - Creation of the output time series, i.e. (index, variable(s)_{t, t + window}, number of patches, resolution, resolution)
        y = np.stack([y[i + window : i + window + window_oxy,  :, :, :] for i in range(t - window - window_oxy)], axis = 0) #.reshape(t - window, (v) * window_oxy, total_patches, resolution, resolution)

        # ------------------------------------------------
        #                   TEMPORAL MODE
        # ------------------------------------------------
        #
        # Documentation
        # -------------
        # In temporal mode, the dataset is split into training, validation and test sets by
        # taking non-overlapping timeseries, i.e. therefore we train on the whole spatial domain
        #
        if mode == "temporal":

            # Computing size of the training, validation and test sets
            training_size, validation_size = int(t * datasets_size[0]), int(t * datasets_size[1])

            # Splitting the dataset into training, validation and test sets while not taking overlapping timeseries
            x_train, x_validation, x_test = x[ : training_size - window, :, :, :, :], x[training_size : training_size + validation_size - window, :, :, :, :], x[training_size + validation_size:, :, :, :, :]
            y_train, y_validation, y_test = y[ : training_size - window, :, :, :, :], y[training_size : training_size + validation_size - window, :, :, :, :], y[training_size + validation_size:, :, :, :, :]

        # ------------------------------------------------
        #                   SPATIAL MODE
        # ------------------------------------------------
        #
        # Documentation
        # -------------
        # In spatial mode, the dataset is split into training, validation and test sets by
        # taking non-overlapping patches, i.e. therefore we train on the whole temporal domain
        #
        if mode == "spatial":

            # Fixing the seed for reproducibility
            np.random.seed(seed = seed)

            # Computing size of the training, validation and test sets
            training_size, validation_size = int(total_patches * datasets_size[0]), int(total_patches * datasets_size[1])

            # Used to randomly permute the patches !
            rand_patches = np.random.permutation(total_patches)

            # Randomly shuffling along the patches axis
            x = x[:, :, rand_patches, :, :]
            y = y[:, :, rand_patches, :, :]

            # Splitting the dataset into training, validation and test sets
            x_train, x_validation, x_test = x[:, :, : training_size, :, :], x[:, :, training_size: training_size + validation_size, :, :], x[:, :, training_size + validation_size :, :, :]
            y_train, y_validation, y_test = y[:, :, : training_size, :, :], y[:, :, training_size: training_size + validation_size, :, :], y[:, :, training_size + validation_size :, :, :]

        # ------------------------------------------------
        #                  PREPROCESSING (2)
        # ------------------------------------------------
        #
        # It is merging time, i.e. final shape (timesteps * patches, variables * window, resolution, resolution)
        self.x_train      = merge_timeseries_and_patches(x_train)
        self.x_validation = merge_timeseries_and_patches(x_validation)
        self.x_test       = merge_timeseries_and_patches(x_test)
        self.y_train      = merge_timeseries_and_patches(y_train)
        self.y_validation = merge_timeseries_and_patches(y_validation)
        self.y_test       = merge_timeseries_and_patches(y_test)

    def get_dataloader(self, type: str, batch_size: int = 32):
        r"""Returns the dataloader for the given type, i.e. training, validation or test"""

        # Security
        assert type in ["train", "validation", "test"], f"ERROR (BlackSea_Dataloader) Type must be either 'train', 'validation' or 'test' ({type})"

        class BS_Dataset(Dataset):
            r"""A simple pytorch dataloader"""

            def __init__(self, x: np.array, y: np.array):
                self.x = x
                self.y = y

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        # Creation of the dataset for dataloader
        dataset = BS_Dataset(getattr(self, f"x_{type}"), getattr(self, f"y_{type}"))

        # Creation of the dataloader
        return DataLoader(dataset, batch_size = batch_size)