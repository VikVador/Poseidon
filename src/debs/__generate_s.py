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
# A simple script to preprocess the data (standardization, flipping, reshaping, ...) the data
#
import os
import json
import wandb
import xarray
import calendar
import numpy as np
from dataset_unp import BlackSea_Dataset_UNPROCESSED

# Determining total number of pixels
#
# Training   : 1980-2015 -> 12784 ---- conversion to total pixels ----> 12784 * 74068 = 946885312
# Validation : 2015-2020 -> 1826  ---- conversion to total pixels ----> 1826  * 74068 = 135248168
# Test       : 2020-2023 -> 1096  ---- conversion to total pixels ----> 1096  * 74068 = 81178528
#
# ----------
# Parameters
# ----------
# Pixels
training_pixels   = 946885312
validation_pixels = 135248168
test_pixels       = 81178528

# Initialization of the project for easy follow-up online
wandb.init(project = "Generating data (Standardized)", mode = "online")

# --------------
# Initialization
# --------------
def preprocess(year_start: int, year_end: int, pixels: int, path: str):
    """ A simple function to preprocess the data """

    # Security
    assert path in ["Training", "Validation", "Test"], print("ERROR PATH")

    # Initialization
    continental_shelf_depth = 150
    hypoxia_treshold        = 63

    # Stores the different useful variables
    t_mean, t_variance, t_standard_deviation = 0, 0, 0 # Temperature
    s_mean, s_variance, s_standard_deviation = 0, 0, 0 # Salinity
    c_mean, c_variance, c_standard_deviation = 0, 0, 0 # Chlorophyll
    h_mean, h_variance, h_standard_deviation = 0, 0, 0 # Sea Surface Height
    o_mean, o_variance, o_standard_deviation = 0, 0, 0 # Oxygen

    # Extracting the Black Sea Mask
    mask = BlackSea_Dataset_UNPROCESSED(1980, 1980, 1, 1).get_mask()

    # ---------- Mean ----------
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):

            # Terminal (1)
            print(f"----- Dataset : {y}-{m} (MEAN)")

            # Tool for loading the datasets
            tool = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

            # Loading the inputs
            data_temperature        = tool.get_data(variable = "temperature",        region = "surface")
            data_salinity           = tool.get_data(variable = "salinity",           region = "surface")
            data_chlorophyll        = tool.get_data(variable = "chlorophyll",        region = "surface")
            data_sea_surface_height = tool.get_data(variable = "height",             region = "surface")
            data_oxygen             = tool.get_data(variable = "oxygen",             region = "bottom")

            # Cliping negative values
            data_oxygen = np.clip(data_oxygen, 0, None)

            # Computing the partial mean (already averaging on total number of pixels in given period)
            o_mean += np.nansum(data_oxygen[:,             mask == 1])/pixels
            t_mean += np.nansum(data_temperature[:,        mask == 1])/pixels
            s_mean += np.nansum(data_salinity[:,           mask == 1])/pixels
            c_mean += np.nansum(data_chlorophyll[:,        mask == 1])/pixels
            h_mean += np.nansum(data_sea_surface_height[:, mask == 1])/pixels

            # WandB (1)
            wandb.log({"Preprocessing (Mean)/Year"  : y,
                       "Preprocessing (Mean)/Month" : m,
                       "Mean/Temperature"           : t_mean,
                       "Mean/Salinity"              : s_mean,
                       "Mean/Chlorophyll"           : c_mean,
                       "Mean/Sea Surface Height"    : h_mean,
                       "Mean/Oxygen"                : o_mean})


    # ---------- Standard Deviation ----------
    for y in range(year_start, year_end + 1):
        for m in range(1, 13):

            # Terminal (2)
            print(f"----- Dataset : {y}-{m} (VARIANCE)")

            # Tool for loading the datasets
            tool = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

            # Loading the inputs
            data_temperature        = tool.get_data(variable = "temperature",        region = "surface")
            data_salinity           = tool.get_data(variable = "salinity",           region = "surface")
            data_chlorophyll        = tool.get_data(variable = "chlorophyll",        region = "surface")
            data_sea_surface_height = tool.get_data(variable = "height",             region = "surface")
            data_oxygen             = tool.get_data(variable = "oxygen",             region = "bottom")

            # Cliping negative values
            data_oxygen = np.clip(data_oxygen, 0, None)

            # Computing the partial variance (already averaging on total number of pixels in given period)
            o_variance += np.nansum( (data_oxygen[:,             mask == 1] - o_mean) ** 2 )/pixels
            t_variance += np.nansum( (data_temperature[:,        mask == 1] - t_mean) ** 2 )/pixels
            s_variance += np.nansum( (data_salinity[:,           mask == 1] - s_mean) ** 2 )/pixels
            c_variance += np.nansum( (data_chlorophyll[:,        mask == 1] - c_mean) ** 2 )/pixels
            h_variance += np.nansum( (data_sea_surface_height[:, mask == 1] - h_mean) ** 2 )/pixels

            # WandB (2)
            wandb.log({"Preprocessing (Variance)/Year"  : y,
                       "Preprocessing (Variance)/Month" : m,
                       "Variance/Temperature"           : t_variance,
                       "Variance/Salinity"              : s_variance,
                       "Variance/Chlorophyll"           : c_variance,
                       "Variance/Sea Surface Height"    : h_variance,
                       "Variance/Oxygen"                : o_variance})

    # Computing the standard deviation
    o_standard_deviation = np.sqrt(o_variance)
    t_standard_deviation = np.sqrt(t_variance)
    s_standard_deviation = np.sqrt(s_variance)
    c_standard_deviation = np.sqrt(c_variance)
    h_standard_deviation = np.sqrt(h_variance)

    # Computing the standardized hypoxia treshold
    hypoxia_treshold_standardized = (hypoxia_treshold - o_mean)/o_standard_deviation

    # ----------- Fixed Variables -----------
    #
    # Used to load other variables
    tool = BlackSea_Dataset_UNPROCESSED(1980, 1980, 1, 1)

    # Loading mask and bathymetry
    mask_continental_shelf = tool.get_mask(depth = continental_shelf_depth)
    bathy_METER            = tool.get_depth(unit = "meter")[0]
    bathy_INDEX            = tool.get_depth(unit = "index")[0]

    # Flipping vertically the data
    mask                   = np.flip(mask,                   axis = 0)
    mask_continental_shelf = np.flip(mask_continental_shelf, axis = 0)
    bathy_METER            = np.flip(bathy_METER,            axis = 0)
    bathy_INDEX            = np.flip(bathy_INDEX,            axis = 0)

    # Removing useless dimensions to become a power of 2
    mask                   = mask[2:, 2:]
    mask_continental_shelf = mask_continental_shelf[2:, 2:]
    bathy_METER            = bathy_METER[2:, 2:]
    bathy_INDEX            = bathy_INDEX[2:, 2:]

    # ------------- Preprocessing the data -------------
    def standardize(data: np.array, mask: np.array, mean: float, std: float, clip: bool = False):
        """Used to process the data easily"""
        data = np.clip(data,  0, None) if clip else data
        data = (data - mean)/std
        data = np.flip(data, axis = 1)
        data = data[:, 2:, 2:]
        data[:, mask == 0] = np.nan
        return data

    for y in range(year_start, year_end + 1):
        for m in range(1, 13):

            # Terminal (3)
            print(f"----- Dataset : {y}-{m} (PREPROCESSING)")

            # Tool for loading the datasets
            tool = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

            # Loading the different inputs
            data_temperature              = tool.get_data(variable = "temperature",  region = "surface")
            data_salinity                 = tool.get_data(variable = "salinity",     region = "surface")
            data_chlorophyll              = tool.get_data(variable = "chlorophyll",  region = "surface")
            data_sea_surface_height       = tool.get_data(variable = "height",       region = "surface")
            data_oxygen                   = tool.get_data(variable = "oxygen",       region = "bottom")
            data_oxygen_continental_shelf = tool.get_data(variable = "oxygen",       region = "bottom", depth = continental_shelf_depth)

            # Standardizing
            data_oxygen                   = standardize(data_oxygen,                                     mask, o_mean, o_standard_deviation, clip = True)
            data_oxygen_continental_shelf = standardize(data_oxygen_continental_shelf, mask_continental_shelf, o_mean, o_standard_deviation, clip = True)
            data_temperature              = standardize(data_temperature,                                mask, t_mean, t_standard_deviation, clip = False)
            data_salinity                 = standardize(data_salinity,                                   mask, s_mean, s_standard_deviation, clip = False)
            data_chlorophyll              = standardize(data_chlorophyll,                                mask, c_mean, c_standard_deviation, clip = False)
            data_sea_surface_height       = standardize(data_sea_surface_height,                         mask, h_mean, h_standard_deviation, clip = False)

            # Datasets - Inputs
            ds_o = xarray.DataArray(data = data_oxygen,             name = "OXY", dims = ["time", "x", "y"], attrs = {"Units": "mmol/m3", "Description": "Oxygen concentration at the bottom layer of the Black Sea"})
            ds_t = xarray.DataArray(data = data_temperature,        name = "TEM", dims = ["time", "x", "y"], attrs = {"Units": "degC",    "Description": "Temperature at the surface of the Black Sea"})
            ds_s = xarray.DataArray(data = data_salinity,           name = "SAL", dims = ["time", "x", "y"], attrs = {"Units": "1e-3",    "Description": "Salinity at the surface of the Black Sea"})
            ds_c = xarray.DataArray(data = data_chlorophyll,        name = "CHL", dims = ["time", "x", "y"], attrs = {"Units": "mmol/m3", "Description": "Chlorophyll concentration at the surface of the Black Sea"})
            ds_h = xarray.DataArray(data = data_sea_surface_height, name = "SSH", dims = ["time", "x", "y"], attrs = {"Units": "[m]",     "Description": "Sea Surface Height of the Black Sea"})

            # Datasets - Input CS
            ds_o_c = xarray.DataArray(data = data_oxygen_continental_shelf, name = "OXYCS", dims = ["time", "x", "y"], attrs = {"Units": "mmol/m3", "Description": "Oxygen concentration at the bottom layer of the Black Sea Continental Shelf"})

            # Datasets - Masks and others
            ds_mask    = xarray.DataArray(data = mask,                   name = "MASK",   dims = ["x", "y"], attrs = {"Description": "Black Sea Mask"})
            ds_mask_cs = xarray.DataArray(data = mask_continental_shelf, name = "MASKCS", dims = ["x", "y"], attrs = {"Description": "Black Sea Continental Shelf Mask"})
            ds_bathy   = xarray.DataArray(data = bathy_METER,            name = "BATHYM", dims = ["x", "y"], attrs = {"Units": "[m]", "Description": "Bathymetry of the Black Sea in meters"})
            ds_bathy_i = xarray.DataArray(data = bathy_INDEX,            name = "BATHYI", dims = ["x", "y"], attrs = {"Units": "[-]", "Description": "Bathymetry of the Black Sea index"})

            # Datasets - Scalars
            ds_hypoxia              = xarray.DataArray(data = hypoxia_treshold,              name = "HYPOXIA",              attrs = {"Description": "Hypoxia Treshold"})
            ds_hypoxia_standardized = xarray.DataArray(data = hypoxia_treshold_standardized, name = "HYPOXIA_STANDARDIZED", attrs = {"Description": "Standardized Hypoxia Treshold"})

            # Datasets - Means and Standard Deviations
            ds_omean = xarray.DataArray(data = o_mean, name = "OMEAN", attrs = {"Description": "Mean Oxygen Concentration"})
            ds_tmean = xarray.DataArray(data = t_mean, name = "TMEAN", attrs = {"Description": "Mean Temperature"})
            ds_smean = xarray.DataArray(data = s_mean, name = "SMEAN", attrs = {"Description": "Mean Salinity"})
            ds_cmean = xarray.DataArray(data = c_mean, name = "CMEAN", attrs = {"Description": "Mean Chlorophyll"})
            ds_hmean = xarray.DataArray(data = h_mean, name = "HMEAN", attrs = {"Description": "Mean Sea Surface Height"})

            ds_ostd = xarray.DataArray(data = o_standard_deviation, name = "OSTD", attrs = {"Description": "Standard Deviation Oxygen Concentration"})
            ds_tstd = xarray.DataArray(data = t_standard_deviation, name = "TSTD", attrs = {"Description": "Standard Deviation Temperature"})
            ds_sstd = xarray.DataArray(data = s_standard_deviation, name = "SSTD", attrs = {"Description": "Standard Deviation Salinity"})
            ds_cstd = xarray.DataArray(data = c_standard_deviation, name = "CSTD", attrs = {"Description": "Standard Deviation Chlorophyll"})
            ds_hstd = xarray.DataArray(data = h_standard_deviation, name = "HSTD", attrs = {"Description": "Standard Deviation Sea Surface Height"})

            # ------- Finalizing --------
            #
            # Creation of the name
            file_name = f"BlackSea-DeepLearning_Standardized_{y}_{m}.nc"

            # Merging everything
            data_final = xarray.merge([ds_o, ds_t, ds_s, ds_c, ds_h, ds_o_c, ds_mask, ds_mask_cs, ds_bathy, ds_bathy_i,
                                       ds_hypoxia, ds_hypoxia_standardized, ds_omean, ds_tmean, ds_smean, ds_cmean, ds_hmean,
                                       ds_ostd, ds_tstd, ds_sstd, ds_cstd, ds_hstd])

            # Adding metadata
            data_final.attrs = {"Author"         : "Victor Mangeleer",
                                "Contact"        : "vmangeleer@uliege.be",
                                "Description"    : "Preprocessed (Standardized) Black Sea Dataset used for Deep Learning",
                                "Dataset"        : f"{path}",
                                "Simulation Date": f"{y}-{m}"}

            # Saving the data
            data_final.to_netcdf(f"../../../../../../../scratch/acad/bsmfc/victor/data/deep_learning/{path}/{file_name}")

            # Displaing information over the terminal
            print(f"Data saved : {file_name}")

            # WandB (3)
            wandb.log({"Preprocessing (Data)/Year"  : y,
                       "Preprocessing (Data)/Month" : m})

# ------
#  Main
# ------
preprocess(year_start = 1980, year_end = 2014, pixels = training_pixels,   path = "Training")
preprocess(year_start = 2015, year_end = 2019, pixels = validation_pixels, path = "Validation")
preprocess(year_start = 2020, year_end = 2022, pixels = test_pixels,       path = "Test")
