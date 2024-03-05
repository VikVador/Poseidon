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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BlackSea_Dataloader():
    r"""A tool to preprocess data and create a dataloader, i.e. used for training, validating and testing the model"""

    def __init__(self, x: list,
                       y: np.array,
                       t: np.array,
                    mesh: np.array,
                    mask: np.array,
         mask_with_depth: np.array,
              bathymetry: np.array,
              window_inp: int   = 1,
              window_out: int   = 1,
          window_transfo: int   = 1,
                    mode: str   = "regression",
        hypoxia_treshold: float = 63,
           datasets_size: list  = [0.7, 0.2]):

        # Concatenating the input
        x = np.stack(x, axis = 1)

        # Retrieiving shape (Ease of comprehension)
        timesteps, variables, _, _ = x.shape

        # Security
        assert window_inp <= int(timesteps/3 - 1),       f"ERROR (BlackSea_Dataloader) Window must be smaller than a third of the input time scale ({window_inp} < {int(timesteps/3 - 1)})"
        assert mode in ["regression", "classification"], f"ERROR (BlackSea_Dataloader) Mode must be either 'regression' or 'classification' ({mode})"

        # Functions
        def splitting(data: np.array, window_input: int, datasets_size: list):
            """Used to split the data into training, validation and test datasets"""

            # Computing sizes for the training, validation and test datasets
            training_size, validation_size = int(data.shape[0] * datasets_size[0]), int(data.shape[0] * datasets_size[1])

            # Splitting the dataset into training, validation and test sets while not taking overlapping timeseries
            return data[0 : training_size - window_input], data[training_size : training_size + validation_size - window_input], data[training_size + validation_size:]

        def rescaling(data: np.array, value : float = None):
            r"""Used to rescale a tensor between [0,1] and, if needed, a given value is also rescaled (useful for oxygen concentration treshold)"""

            # Determining the minimum and maximum values
            min_value = np.nanmin(data)
            max_value = np.nanmax(data)

            # Shift the data to ensure minimum value is 0
            shifted_data = data - min_value

            # Normaliinge the data
            normalized_data = shifted_data / (max_value - min_value)

            # If a value is given, rescale it
            return normalized_data, (value - min_value)/(max_value - min_value) if value is not None else None

        def spatialize(data: np.array, mesh: np.array, bathymetry: np.array):
            """Used to add spatial information to the data, i.e. bathymetry and mesh"""

            # Retrieiving the dimensions
            timesteps, days, metrics, variables, x_res, y_res = data.shape

            # Adding the missing dimensions
            mesh       = np.expand_dims(mesh,       axis = (0, 1, 2))
            bathymetry = np.expand_dims(bathymetry, axis = (0, 1, 2))

            # Replicating the mesh and bathymetry to match each dimension
            for i, axe in zip(range(3), [timesteps, days, metrics]):
                mesh       = np.repeat(mesh,       repeats = axe, axis = i)
                bathymetry = np.repeat(bathymetry, repeats = axe, axis = i)

            # Concatenating everything to the original data
            return np.concatenate([data, mesh, bathymetry], axis = 3)

        def formulate(mode: str, data: np.array, mask: np.array, treshold: float):
            """Used to formulate the problem, i.e. classification or regression by transforming the output"""

            # Retrieiving shape (Ease of comprehension)
            dimensions = data.shape

            # Classification, i.e. the emulator predicts wether or not a region (pixel) is in hypoxia
            if mode == "classification":

                # Retrieving indices for regions with or w/o hypoxia
                indexes_hypoxia, indexes_oxygenated = y <= self.normalized_deoxygenation_treshold, self.normalized_deoxygenation_treshold < y

                # Adding new axes for classes, i.e. introducing channel for probabilities per class (t, f, x, y) to (t, f, number of classes (= 2), x, y)
                data = np.repeat(np.expand_dims(data, axis = 2), repeats = 2, axis = 2)

                # Adding class for each pixel (c = 0 : oxygenated (no hypoxia), c = 1 == hypoxia)
                for c in range(2):

                    # Extraction of the class
                    data_class = data[:, :, c, :, :]

                    # --- Class 0 (Oxygenated) ---
                    if c == 0:
                        data_class[indexes_hypoxia]    = 0
                        data_class[indexes_oxygenated] = 1

                    # --- Class 1 (Hypoxia) ---
                    else:
                        data_class[indexes_hypoxia]    = 1
                        data_class[indexes_oxygenated] = 0

                # Masking the land
                data[:, :, :, mask == 0] = -1

            # Regression, i.e. the emulator predicts the oxygen concentration
            else:

                # Masking the land
                data[:, :, mask == 0] = -1

                # Adding new axis for the concentration valu
                data = np.expand_dims(data, axis = 2)


            # Returning the formulated data
            return data

        def transformations(data: np.array, time: np.array, window_transformation: int):
            """Used to transform the data, i.e. mean, variance and median values over a given window transformation"""

            # Retrieiving shape (Ease of comprehension)
            timesteps, values, variables, x_res, y_res = data.shape

            # Security
            assert window_transformation <= timesteps/2, f"ERROR (BlackSea_Dataloader) Window transformation must be smaller than half the input time scale ({window_transformation} < {timesteps/2})"

            # Checking if transformations are needed, i.e. if the window_transformation is > 1 than we can transform the original data
            if window_transformation == 1:
                return np.expand_dims(data, axis = 2), np.expand_dims(time, axis = 2)

            # Stores the transformed data and time
            transformed_data, transformed_time = list(), list()

            # Number of batches one can create
            n_batches = int(np.ceil(values/window_transformation))

            # Creating the batches
            for b in range(n_batches):

                # Starting and ending indexes to extract the batch
                batch_start = b * window_transformation
                batch_end   = b * window_transformation + window_transformation

                # Stores temporarily the batch transformed data
                batch_data = list()

                # Extracting the batches
                data_batch = data[:, batch_start : batch_end, :, :, :]

                # Performing the transformation(s), i.e. mean, variance and median values.
                batch_data.append(np.mean(   data_batch, axis = 1))
                batch_data.append(np.var(    data_batch, axis = 1))
                batch_data.append(np.median( data_batch, axis = 1))

                # Transforming the batches into one numpy array
                transformed_data.append(np.stack(batch_data, axis = 1))

                # Extracting the corresponding time slices, i.e. days IDs to know which days have been used to compute mean, ...
                data_time = time[:, batch_start : batch_end]

                # Checks if the time slice is smaller than the window transformation, i.e. we need to fill the missing days with 0
                missing_days = window_transformation - len(data_time[0])

                # Checks if missing days are present
                if 0 < missing_days:

                    # Storing the new time slice
                    data_time_corrected = list()

                    # Fixing the time slices
                    for d in data_time:
                        data_time_corrected.append(list(d) + [0 for i in range(missing_days)])

                    # Addingt the (fixed) time
                    transformed_time.append(data_time_corrected)

                # Everything is fine, i.e. each metrics were computed using the same number of days !
                else:

                    # Adding the time
                    transformed_time.append(data_time)

            # Finalizing the transformations
            return np.stack(transformed_data, axis = 1), np.stack(transformed_time, axis = 1)

        # ------------------------------------------------
        #                  PREPROCESSING
        # ------------------------------------------------
        #
        # Rescaling the outpout and storing the normalized concentration treshold
        y, self.normalized_deoxygenation_treshold = rescaling(y, hypoxia_treshold)

        # Rescaling the input(s)
        for v in range(variables):
            x[:, v, :, :], _ = rescaling(x[:, v, :, :])

        # Total number of time series input/output pairs
        n_samples = timesteps - window_inp - window_out

        # Generating time series
        x = np.stack([x[i : i + window_inp, :, :, :]                                for i in range(n_samples)], axis = 0)
        t = np.stack([t[i : i + window_inp]                                         for i in range(n_samples)], axis = 0)
        y = np.stack([y[i + window_inp - 1 : i + window_inp + window_out - 1, :, :] for i in range(n_samples)], axis = 0)

        # Transforming the inputs into mean, variance and median over window transformations days
        x, t = transformations(data = x, time = t, window_transformation = window_transfo)

        # Hidding the land and the regions of no interest
        x[:, :, :, :, mask == 0] = -1

        # Adding spatial information, i.e. bathymetry and mesh
        x = spatialize(data = x, mesh = mesh, bathymetry = bathymetry)

        # Formulating the problem, i.e. classification or regression by transforming the output
        y = formulate(mode = mode,
                      data = y,
                      mask = mask_with_depth,
                  treshold = self.normalized_deoxygenation_treshold)

        # Creating the training, validation and test datasets
        self.x_train, self.x_validation, self.x_test = splitting(data = x, window_input = window_inp, datasets_size = datasets_size)
        self.t_train, self.t_validation, self.t_test = splitting(data = t, window_input = window_inp, datasets_size = datasets_size)
        self.y_train, self.y_validation, self.y_test = splitting(data = y, window_input = window_inp, datasets_size = datasets_size)

        # Stores the number of samples of each datasets
        self.number_training_samples   = self.x_train.shape[0]
        self.number_validation_samples = self.x_validation.shape[0]
        self.number_test_samples       = self.x_test.shape[0]

    def get_number_of_samples(self, type : str):
        r"""Returns the number of samples in a given dataset, i.e. training, validation or test"""
        return getattr(self, f"number_{type}_samples")

    def get_number_of_batches(self, type: str, batch_size: int = 64):
        r"""Returns the number of batches in a given dataset, i.e. training, validation or test"""
        return int(getattr(self, f"x_{type}").shape[0] // batch_size + 1)

    def get_normalized_deoxygenation_treshold(self):
        r"""Used to retreive the normalized deoxygenation treshold"""
        return self.normalized_deoxygenation_treshold

    def get_dataloader(self, type: str, batch_size: int = 64):
        r"""Creates and returns a dataloader for a given type, i.e. training, validation or test"""

        # Security
        assert type in ["train", "validation", "test"], f"ERROR (BlackSea_Dataloader) Type must be either 'train', 'validation' or 'test' ({type})"

        class BS_Dataset(Dataset):
            r"""Pytorch dataloader"""

            def __init__(self, x: np.array, t: np.array, y: np.array):
                self.x = x
                self.t = t
                self.y = y

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, idx):
                return self.x[idx], self.t[idx], self.y[idx]

        # Creation of the dataset for dataloader
        dataset = BS_Dataset(x = getattr(self, f"x_{type}"),
                             t = getattr(self, f"t_{type}"),
                             y = getattr(self, f"y_{type}"))

        # Creation of the dataloader
        return DataLoader(dataset, batch_size = batch_size)
