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
# A tool to load raw Black Sea datasets coming from the NEMO simulator.
#
import os
import json
import xarray
import calendar
import numpy as np

from tools import get_data_info, get_mask_info


class BlackSea_Dataset():
    r"""A simple tool to load data of Black Sea simulations (NEMO Simulator) from 1980 to 2023."""

    def __init__(self, year_start: int = 1980,
                         year_end: int = 1980,
                      month_start: int = 1,
                        month_end: int = 1,
                           folder: str = "output_HR001"):
        super().__init__()

        # Security (1)
        assert year_start  in [i for i in range(1980, 2023)], f"ERROR (Dataset, init) - Incorrect starting year (1980 <= {year_start} <= 2022)"
        assert year_end    in [i for i in range(1980, 2023)], f"ERROR (Dataset, init) - Incorrect ending year (1980 <= {year_end} <= 2022)"
        assert month_start in [i for i in range(1, 13)],      f"ERROR (Dataset, init) - Incorrect starting month (1 <= {month_start} <= 12)"
        assert month_end   in [i for i in range(1, 13)],      f"ERROR (Dataset, init) - Incorrect ending month (1 <= {month_end} <= 12)"
        assert  year_start <= year_end,                       f"ERROR (Dataset, init) - Incorrect years ({year_start} <= {year_end})"

        # Loading the name of all the useless variables, i.e. not usefull for our specific problem (for the sake of efficiency)
        with open('../../information/useless.txt', 'r') as file:
            self.useless_variables = json.load(file)

        # Retrieving possible path to the folder containing the datasets
        data_path_cluster, data_path_local = get_data_info()

        # Path to the folder containing the data
        self.datasets_folder = data_path_cluster if os.path.exists(data_path_cluster) else \
                               data_path_local

        # Stores all the relevant paths datasets
        paths = list()

        # Extraction
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):

                # Security (Start and ending years)
                if year == year_start:
                    if month < month_start:
                        continue
                if year == year_end:
                    if month_end < month:
                        break

                # Adding the paths
                paths = paths + [self.datasets_folder + f"BlackSea-DeepLearning_V2_{year}_{month}.nc"]

        # Saving other relevant information
        self.paths       = paths
        self.month_start = month_start
        self.month_end   = month_end
        self.year_end    = year_end
        self.year_start  = year_start
        self.folder      = folder

    def get_mesh(self, x: int, y: int):
        r"""Used to retrieve a mesh with normalized coordinates for the given shape (x, y)"""

        # Creation of the mesh
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, 1, num = x), np.linspace(0, 1, num = y), indexing = 'ij')

        # Concatenation of the mesh (np.float32 is the type needed for torch when converted afterforwards)
        return np.stack((x_mesh, y_mesh), axis = 0, dtype = np.float32)

    def get_depth(self, unit: str):
        """Used to retrieve the maximum depths position (indexes in 3D data) or the maximum depths values (in meters)"""

        # Security
        assert unit in ["index", "meter"], f"ERROR (get_depth), Incorrect unit ({unit})"

        # Loading the data
        data = xarray.open_dataset(self.paths[0])

        # Returning the corresponding mask
        return data["BATHYM"].data if unit == "meter" else data["BATHYI"].data

    def get_mask(self, continental_shelf: bool = False):
        r"""Used to retreive a mask of the Black Sea, i.e. 0 if land, 1 if the Black Sea. If depth is given, it will also set to 0 all regions below that depth"""

        # Loading the data
        data = xarray.open_dataset(self.paths[0])

        # Returning the corresponding mask
        return data["MASK"].data if continental_shelf == False else data["MASKCS"].data

    def get_days(self):
        """Used to get the IDs of days for a given time period, i.e. for each sample we have its day ID (1 to 365, repeated if multiple years are given)"""

        def is_leap_year(year):
            """Used to check if a given year is a leap year"""
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        # Stores the day IDs
        day_ids = []

        # Looping over the years
        for year in range(self.year_start, self.year_end + 1):

            # Handling the starting and ending months
            start_month = self.month_start if year == self.year_start else 1
            end_month   = self.month_end   if year == self.year_end   else 12

            # Used to keep track of IDs
            current_day_id = 1

            # Looping over the months
            for month in range(1, end_month + 1):

                # Determine the number of days
                _, num_days = calendar.monthrange(year, month)

                # Append day IDs for the current month
                day_ids.extend(range(current_day_id, current_day_id + num_days)) if start_month <= month else None

                # Increment current day ID
                current_day_id += num_days

                # Correction for leaping years
                if month == 2:
                    if not is_leap_year(year):
                        current_day_id += 1

        return np.array(day_ids, dtype = np.float32)

    def get_data(self, variable: str):
        r"""Used to retreive the data for a given variable"""

        # Security (1)
        assert variable in ["temperature", "salinity", "oxygen", "chlorophyll", "kshort", "klong"], f"ERROR (get_data), Incorrect variable ({variable})"

        def translate(variable: str):
            r"""Used to translate a variable into its name in the dataset, retrieve the type of dataset and the useless variables (the other ones)"""

            # Stores the translations for all the EO variables and oxygen
            translations = {"temperature" : "TEMP",
                            "salinity"    : "SAL",
                            "oxygen"      : "OXYCS",
                            "oxygenall"   : "OXY",
                            "chlorophyll" : "CHL",
                            "kshort"      : "KSHORT",
                            "klong"       : "KLONG"}

            # Retrieving translation
            return translations[variable]

        # Translation
        variable = translate(variable)

        # Opening all the datasets
        data = xarray.open_mfdataset(self.paths, combine='nested', concat_dim = "time")

        # Returns the corresponding data
        return np.array(data[variable].data)