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
import numpy as np
from tools import *


class BlackSea_Dataset():
    r"""A simple tool to load data of Black Sea simulations (NEMO Simulator) from 1980 to 2023."""

    def __init__(self, year_start: int = 1980,
                         year_end: int = 1980,
                      month_start: int = 1,
                        month_end: int = 1,
                           folder: str = "output_HR001"):
        super().__init__()

        # Security (1)
        assert year_start  in [i for i in range(1980, 2023)], f"ERROR (Dataset, init) - Incorrect starting year ({year_start})"
        assert year_end    in [i for i in range(1980, 2023)], f"ERROR (Dataset, init) - Incorrect ending year ({year_end})"
        assert month_start in [i for i in range(1, 13)],      f"ERROR (Dataset, init) - Incorrect starting month ({month_start})"
        assert month_start in [i for i in range(1, 13)],      f"ERROR (Dataset, init) - Incorrect ending month ({month_end})"
        assert year_start <= year_end,                        f"ERROR (Dataset, init) - Incorrect years ({year_start} <= {year_end})"

        # Loading the name of all the useless variables, i.e. not usefull for our specific problem (for the sake of efficiency)
        with open('../../information/useless.txt', 'r') as file:
            self.useless_variables = json.load(file)

        # Loading (all) the dictionnaries containing the path to each dataset, i.e. "YEAR-MONTH : PATH(S)"
        with open('../../information/grid_T.txt', 'r') as file:
            paths_physics_datasets_all = json.load(file)

        with open('../../information/ptrc_T.txt', 'r') as file:
            paths_biogeochemistry_datasets_all = json.load(file)

        # Retrieving possible path to the folder containing the datasets
        data_path_cluster, data_path_local = get_data_info()

        # Path to the folder containing the data
        self.datasets_folder = data_path_cluster + f"{folder}/" if os.path.exists(data_path_cluster) else \
                               data_path_local   + f"{folder}/"

        # Stores all the relevant paths datasets
        self.paths_physics_datasets, self.paths_biogeochemistry_datasets = list(), list()

        # Extraction
        for year in range(year_start, year_end + 1):
            for month in range(month_start, month_end + 1):

                # Converting to strings
                year  = str(year)
                month = f"0{month}" if month < 10 else str(month)

                # Creation of the key
                key = f"{year}-{month}"

                # Retreiving the paths
                self.paths_physics_datasets         += [self.datasets_folder + p for p in paths_physics_datasets_all[key]]
                self.paths_biogeochemistry_datasets += [self.datasets_folder + p for p in paths_biogeochemistry_datasets_all[key]]

        # Saving other relevant information
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
        assert unit in ["index", "meter"], f"ERROR (get_depths), Incorrect type ({unit})"

        # Path to the mesh file location
        path_mesh = get_mesh_path()

        # Loading the information in the correct shape
        return xarray.open_dataset(path_mesh, engine = "h5netcdf").bathy_metry.data.astype('float32') if unit == "meter" else \
               xarray.open_dataset(path_mesh, engine = "h5netcdf").mbathy.data

    def get_mask(self, depth: int = None):
        r"""Used to retreive a mask of the Black Sea, i.e. 0 if land, 1 if the Black Sea. If depth is given, it will also set to 0 all regions below that depth"""

        # Retrieving possible path to the data
        mask_path_cluster, mask_path_local, mask_name = get_mask_info()

        # Loading the dataset containing information about the Black Sea mesh
        mesh_data = xarray.open_dataset(mask_path_cluster + mask_name if os.path.exists(mask_path_cluster) else \
                                        mask_path_local   + mask_name,
                                        engine = "h5netcdf")

        # Loading the complete Black sea mask
        bs_mask = mesh_data.tmask[0, 0].data

        # Checks if we want to hide regions below a given depth
        if not depth == None:

            # Retreives the bottom depth in [m] for each pixel
            depth_values = mesh_data.bathy_metry.data[0]

            # Remove all information for regions located below the given depth
            bs_mask[depth <= depth_values] = 0

        # Returning the processed mask
        return bs_mask

    def get_data(self, variable: str,
                          level: int = None,
                         region: str = None,
                          depth: int = None):
        r"""Used to retreive the data for a given variable at a specific level or a specific region"""

        # Security (1)
        assert variable in ["temperature", "salinity", "oxygen", "chlorophyll", "kshort", "klong"], f"ERROR (get_data), Incorrect variable ({variable})"
        assert level  == None or (0 <= level and level < 59),                                       f"ERROR (get_data), Incorrect level (0 < {level} < 59)"
        assert region == None or region in ["surface", "bottom", "all"],                            f"ERROR (get_data), Incorrect region ({region})"

        # Security (2) - Choosing between level or region
        assert not (level == None and region == None), f"ERROR (get_data), You must specify either a level or a region"
        assert not (level != None and region != None), f"ERROR (get_data), You must specify either a level or a region, not both"

        def translate(variable: str):
            r"""Used to translate a variable into its name in the dataset, retrieve the type of dataset and the useless variables (the other ones)"""

            # Stores the translations for all the EO variables and oxygen
            translations = {"temperature" : ["votemper", "physics"],
                            "salinity"    : ["vosaline", "physics"],
                            "oxygen"      : ["DOX",      "biogeochemistry"],
                            "chlorophyll" : ["CHL",      "biogeochemistry"],
                            "kshort"      : ["KBIOS",    "biogeochemistry"],
                            "klong"       : ["KBIOL",    "biogeochemistry"]}

            # Retrieving translation
            v, v_type = translations[variable]

            # Returns also the useless variables
            return v, v_type, [values[0] for key, values in translations.items() if key != variable]

        def get_bottom(data: np.array, depth = None):
            r"""Used to retreive the data profile (2D) everywhere at the bottom of the Black Sea (None) of for all regions above a given depth"""

            # Security
            assert len(data.shape) == 4, f"ERROR (get_bottom), Incorrect data shape ({data.shape}), i.e. input dimensions should be (time, depth, y, x)"

            # Retreiving the bathymetry mask b(t, x, y) = z_bottom, i.e. index at which we found bottom of the sea
            bathy_mask = self.get_bathymetry()

            # Creation of x and y indexes to make manipulation
            x, y = np.arange(bathy_mask.shape[2]), np.arange(bathy_mask.shape[1])
            xidx = x.reshape(-1,1).repeat(len(y), axis = 1).T
            yidx = y.reshape(-1,1).repeat(len(x), axis = 1)

            # Retreiving the data everywhere at the bottom
            data = data[:, bathy_mask[0] - 1, yidx, xidx]

            # Hiding the regions below the given depth
            if not depth == None:
                data[:, self.get_mask(depth = depth) == 0] = np.nan

            return data

        # Translation
        variable, variable_type, other_useless_variables = translate(variable)

        # Loading the data (3D field)
        data = xarray.open_mfdataset(self.paths_physics_datasets if variable_type == "physics" else self.paths_biogeochemistry_datasets,
                                     engine         = "h5netcdf",
                                     parallel       = True,
                                     drop_variables = self.useless_variables + other_useless_variables)

        # Level, i.e. selecting a specific depth by its index
        if not level == None:
            return data[variable][:, level, :, :].data.compute()

        # All (3D)
        if region == "all":
            return data[variable][:, :, :, :].data.compute()

        # Surface (2D)
        if region == "surface":
            return data[variable][:, 0, :, :].data.compute()

        # Bottom (2D)
        if region == "bottom":
            return get_bottom(data[variable].data.compute(), depth = depth)
