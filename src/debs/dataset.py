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
# A tool to load training (1980 to 2014), validation (2015 to 2019) and test (2020 to 2022) standardized Black Sea datasets coming from the NEMO simulator.
#
import os
import xarray
import numpy as np


class BlackSea_Dataset():
    r"""A simple tool to load data of Black Sea simulations (NEMO Simulator)"""

    def __init__(self, dataset_type: str = "Training"):

        # Security
        assert dataset_type in ["Training", "Validation", "Test"], f"ERROR (BlackSea_Dataset), Incorrect dataset type ({dataset_type})"

        # Path to cluster files
        path = f"../../../../../../../scratch/acad/bsmfc/victor/data/deep_learning/{dataset_type}/"

        # Creation of the files name
        month_start, month_end, year_start, year_end = 1, 12, 0, 0

        if dataset_type == "Training":
            year_start, year_end = 1980, 2014
        elif dataset_type == "Validation":
            year_start, year_end = 2015, 2019
        else:
            year_start, year_end = 2020, 2022

        files = [path + f"BlackSea-DeepLearning_Standardized_{y}_{m}.nc" for y in range(year_start, year_end + 1) for m in range(1, 13)]

        # Loading the data and saving other relevant information
        self.data        = xarray.open_mfdataset(files, combine = 'nested', concat_dim = "time").compute()
        self.data_type   = dataset_type
        self.month_start = month_start
        self.month_end   = month_end
        self.year_start  = year_start
        self.year_end    = year_end

    def get_data(self, variable: str):
        """ Used to access the input/output data more clearly"""

        # Security
        assert variable in ["temperature", "salinity", "chlorophyll", "height", "oxygen"], f"ERROR (get_data), Incorrect variable ({variable})"

        # Stores the translations
        translations = {"temperature": "TEM", "salinity": "SAL", "chlorophyll": "CHL", "height": "SSH", "oxygen": "OXYCS"}

        # Extracting the data
        return self.data[translations[variable]].data

    def get_mesh(self, x: int = 256, y: int = 576):
        r"""Used to retrieve a mesh with normalized coordinates for the given shape (x, y)"""

        # Creation of the mesh
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, num = x), np.linspace(0, 1, num = y), indexing = 'ij')

        # Concatenation of the mesh (np.float32 is the type needed for torch when converted afterforwards)
        return np.stack((x_mesh, y_mesh), axis = 0, dtype = np.float32)

    def get_depth(self, unit: str = "index"):
        """Used to retrieve the maximum depths position (indexes in 3D data) or the maximum depths values (in meters)"""

        # Security
        assert unit in ["index", "meter"], f"ERROR (get_depth), Incorrect unit ({unit})"

        # Loading the data
        depths = self.data["BATHYM"].data[0] if unit == "meter" else self.data["BATHYI"].data[0]

        # Returning with an additional dimension
        return np.expand_dims(depths, axis = 0)

    def get_mask(self, continental_shelf: bool = False):
        r"""Used to retreive a mask of the Black Sea, i.e. 0 if land, 1 if the Black Sea. If depth is given, it will also set to 0 all regions below that depth"""

        # Loading the data
        mask = self.data["MASK"].data[0] if continental_shelf == False else self.data["MASKCS"].data[0]

        # Returning with an additional dimension
        return np.expand_dims(mask, axis = 0)

    def get_treshold(self, standardized = False):
        r"""Used to retrieve the hypoxia treshold value, i.e. the oxygen concentration below which hypoxia is considered to occur"""
        return self.data["HYPOXIA_STANDARDIZED"].data[0] if standardized else self.data["HYPOXIA"].data[0]

    def get_mean(self, variable: str):
        """Used to retreive the mean value of a variable for a given dataset type"""

        # Security
        assert variable in ["temperature", "salinity", "chlorophyll", "height", "oxygen"], f"ERROR (get_data), Incorrect variable ({variable})"

        # Stores the translations
        translations = {"temperature": "TMEAN", "salinity": "SMEAN", "chlorophyll": "CMEAN", "height": "HMEAN", "oxygen": "OMEAN"}

        # Extracting the data
        return self.data[translations[variable]].data[0]

    def get_standard_deviation(self, variable: str):
        """Used to retreive the stand value of a variable for a given dataset type"""

        # Security
        assert variable in ["temperature", "salinity", "chlorophyll", "height", "oxygen"], f"ERROR (get_data), Incorrect variable ({variable})"

        # Stores the translations
        translations = {"temperature": "TSTD", "salinity": "SSTD", "chlorophyll": "CSTD", "height": "HSTD", "oxygen": "OSTD"}

        # Extracting the data
        return self.data[translations[variable]].data[0]

    def get_time(self):
        """Generate time information about the dataset, i.e. for each day, retrieves the relative index, the month and year"""

        # Used to play easily with dates
        from datetime import datetime, timedelta

        # Helper functions
        def get_index_day(date):
            """Get the index of the day in the year"""
            return date.timetuple().tm_yday

        def get_index_month(date):
            """Get the index of the month in the year"""
            return date.month

        def get_index_year(date):
            """Get the year associated with the given date"""
            return date.year

        # Conversion of the date to appropriate format
        start_date = datetime(self.year_start, 1,   1)
        end_date   = datetime(self.year_end,  12,  31)

        # Creation of the dates
        num_days    = (end_date - start_date).days + 1
        dates       = [start_date + timedelta(days = i) for i in range(num_days)]
        index_day   = np.array([get_index_day(date)   for date in dates])
        index_month = np.array([get_index_month(date) for date in dates])
        index_year  = np.array([get_index_year(date)  for date in dates])

        return index_day, index_month, index_year