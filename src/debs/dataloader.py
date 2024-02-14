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
# A tool to preprocess data and create a dataloader, i.e. used for training, validating and testing the model
#
import numpy as np

# Torch
import torch
from   torch.utils.data import Dataset
from   torch.utils.data import DataLoader


class BlackSea_Dataloader():
    r"""A tool to preprocess data and create a dataloader, i.e. used for training, validating and testing the model"""

    def __init__(self, x: list,
                       y: np.array,
                 bs_mask: np.array,
      bs_mask_with_depth: np.array,
              window_inp: int   = 1,
              window_out: int   = 1,
                   mode : str   = "regression",
        hypoxia_treshold: float = 63,
           datasets_size: list  = [0.7, 0.2],
                    seed: int   = 2701):

        # ------------------------------------------------
        #                  PREPROCESSING (1)
        # ------------------------------------------------
        #
        def compute_normalized_deoxygenation_treshold(data: np.array, hypoxia_treshold: float):
            r"""Used to retreive the normalized deoxygenation treshold, i.e. if we need the regions with
                oxygen concentrations above 63 [mmol/m3] we need the normalized value of 63 [mmol/m3]"""

            # Determine the minimum and maximum values of the data
            min_value = np.nanmin(data)
            max_value = np.nanmax(data)

            # Computes the normalized hypoxia treshold value
            return (hypoxia_treshold - min_value) / (max_value - min_value)

        def preprocess(data: np.array, bs_mask: np.array):
            r"""Used to mask the NaNs as well as the land and perform a min/max normalization on the data"""

            # Determine the minimum and maximum values of the data
            min_value = np.nanmin(data)
            max_value = np.nanmax(data)

            # Rescale the data to ensure non-negative values
            data = data + np.abs(min_value) if min_value < 0 else data

            # Normalizing the data
            return (data - min_value) / (max_value - min_value)

        # Stores the normalized concentration treshold
        self.normalized_deoxygenation_treshold = compute_normalized_deoxygenation_treshold(y, hypoxia_treshold)

        # Concatenation of inputs and output
        x = np.stack([y] + x, axis = 1)

        # Reshaping dimensions to be a power of 2
        x                  = x[:, :, :-2, :-2]
        bs_mask            = bs_mask[:-2, :-2]
        bs_mask_with_depth = bs_mask_with_depth[:-2, :-2]

        # Current shape of the input (ease of comprehension)
        t, v, x_res, y_res = x.shape

        # Security
        assert window_inp <= int(t/3 - 1),               f"ERROR (BlackSea_Dataloader) Window must be smaller than a third of the input time scale ({window_inp} < {int(t/3 - 1)})"
        assert mode in ["regression", "classification"], f"ERROR (BlackSea_Dataloader) Mode must be either 'regression' or 'classification' ({mode})"

        # Preprocessing of all the data
        for i in range(v):
            x[:, i, :, :] = preprocess(x[:, i, :, :], bs_mask)

        # Separation of the x and y data (to avoid a mess)
        y = x[:, 0,  :, :]
        x = x[:, 1:, :, :]

        # Input - Creation of the input time series, i.e. (sample, variable(s)_{t, t + window}, resolution, resolution)
        x = np.stack([x[i : i + window_inp, :, :, :] for i in range(t - window_inp - window_out)], axis = 0).reshape(t - window_inp - window_out, (v - 1) * window_inp, x_res, y_res)

        # Output - Creation of the output time series, i.e. (sample, variable_{t, t + window}, resolution, resolution)
        y = np.stack([y[i + window_inp - 1 : i + window_inp + window_out - 1, :, :] for i in range(t - window_inp - window_out)], axis = 0)

        # ------------------------------------------------
        #              PROBLEM FORMULATION
        # ------------------------------------------------
        #
        # Classification, i.e. the emulator predicts wether or not a region (pixel) is in hypoxia
        if mode == "classification":

            # Current shape of the input (ease of comprehension)
            t, forecasted_days, x_res, y_res = x.shape

            # Retrieving indices for regions with or w/o hypoxia
            index_hypoxia    = y  < self.normalized_deoxygenation_treshold
            index_no_hypoxia = y >= self.normalized_deoxygenation_treshold

            # Adding new axes and reshaping, i.e (t, f, x, y) to (t, f, number of classes (= 2), x, y)
            y = np.expand_dims(y, axis = 2)
            y = np.repeat(y, 2, axis = 2)

            # Adding class for each pixel (c = 0 == no hypoxia, c = 1 == hypoxia)
            for i in range(2):

                # Current class
                class_oxy = y[:, :, i, :, :]

                # 1 - Class 0, No-Hypoxia
                if i == 0:
                    class_oxy[index_hypoxia]    = 0
                    class_oxy[index_no_hypoxia] = 1

                # 2 - Class 1, Hypoxia
                else:
                    class_oxy[index_hypoxia]    = 1
                    class_oxy[index_no_hypoxia] = 0

            # Output - Hidding the land and the regions of no interest, i.e. below depth treshold (extra dimension)
            y[:, :, :, bs_mask_with_depth == 0] = -1

        # Regression, i.e. the emulator predicts the oxygen concentration
        else:

            # Output - Hidding the land and the regions of no interest, i.e. below depth treshold
            y[:, :, bs_mask_with_depth == 0] = -1

        # Input - Hidding the land and the regions of no interest, i.e. below depth treshold
        x[:, :,  bs_mask == 0] = -1

        # ------------------------------------------------
        #              PREPROCESSING (2)
        # ------------------------------------------------
        #
        # Computing size of the training, validation and test sets
        training_size, validation_size = int(t * datasets_size[0]), int(t * datasets_size[1])

        # Splitting the dataset into training, validation and test sets while not taking overlapping timeseries
        self.x_train, self.x_validation, self.x_test = x[ : training_size - window_inp, :, :, :], x[training_size : training_size + validation_size - window_inp, :, :, :], x[training_size + validation_size:, :, :, :]
        self.y_train, self.y_validation, self.y_test = y[ : training_size - window_inp, :, :, :], y[training_size : training_size + validation_size - window_inp, :, :, :], y[training_size + validation_size:, :, :, :]

    def get_normalized_deoxygenation_treshold(self):
        r"""Used to retreive the normalized deoxygenation treshold"""
        return self.normalized_deoxygenation_treshold

    def get_number_of_batches(self, type: str, batch_size: int = 64):
        r"""Returns the number of batches in a given dataset, i.e. training, validation or test"""
        return int(getattr(self, f"x_{type}").shape[0] // batch_size + 1)

    def get_dataloader(self, type: str, bathy : torch.Tensor = None, mesh : torch.Tensor = None, batch_size: int = 64):
        r"""Creates and returns a dataloader for a given type, i.e. training, validation or test"""

        # Security
        assert type in ["train", "validation", "test"], f"ERROR (BlackSea_Dataloader) Type must be either 'train', 'validation' or 'test' ({type})"

        class BS_Dataset(Dataset):
            r"""A custom pytorch dataloader"""

            def __init__(self, x: np.array, y: np.array, bathy : np.array = None, mesh : np.array = None):

                # Storing info for the dataloader (reshaping bathymetry to be multiple of 2)
                self.x     = x
                self.y     = y
                self.mesh  = mesh

                # Reshaping bathymetry to be multiple of 2
                self.bathy = torch.from_numpy(bathy[:, :-2, :-2]) if bathy is not None else None

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):

                # Extracting the sample
                xi = self.x[idx]

                # Adding bathymetry if needed
                xi = np.concatenate((xi, self.bathy), axis = 0) if self.bathy is not None else xi

                # Adding the mesh
                xi = np.concatenate((xi, self.mesh), axis = 0) if self.mesh is not None else xi

                return xi, self.y[idx]

        # Creation of the dataset for dataloader
        dataset = BS_Dataset(x = getattr(self, f"x_{type}"), y = getattr(self, f"y_{type}"), bathy = bathy, mesh = mesh)

        # Creation of the dataloader
        return DataLoader(dataset, batch_size = batch_size)