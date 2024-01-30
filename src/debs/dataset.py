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
import xarray
import numpy as np

# Custom library
from tools import get_data_path, get_mesh_path

class BlackSea_Dataset():
    r"""A simple tool to load data for the Black Sea dataset (NEMO Simulator)."""

    def __init__(self, year_start: int, year_end: int, month_start: int, month_end: int, variable: str, folder:str = "output_HR004"):
        super().__init__()

        # Security
        assert year_start  in [i for i in range(10)], f"ERROR (Dataset, init) - Incorrect starting year ({year_start})"
        assert year_end    in [i for i in range(10)], f"ERROR (Dataset, init) - Incorrect ending year ({year_end})"
        assert month_start in [i for i in range(13)], f"ERROR (Dataset, init) - Incorrect starting month ({month_start})"
        assert month_start in [i for i in range(13)], f"ERROR (Dataset, init) - Incorrect ending month ({month_end})"
        assert year_start <= year_end,                f"ERROR (Dataset, init) - Incorrect years ({year_start} <= {year_end})"
        assert variable in ["grid_U", "grid_V", "grid_W", "grid_T", "ptrc_T", "btrc_T"], f"ERROR (Dataset, init) - Incorrect variable ({variable})"
        if year_start == year_end:
            assert month_start <= month_end, f" ERROR (Dataset, init) - Incorrect months ({month_start} <= {month_end})"

        # Stores a list of useless variables, i.e. not usefull for our specific problem (for the sake of efficiency)
        self.useless_variables = ['time_centered',
                                  'deptht_bounds',
                                  'time_centered_bounds',
                                  'time_counter_bounds',
                                  'time_instant_bounds',
                                  'ssh', 'mldkz5', 'mldr10_1', 'mld_bs', 'rho', 'CCC', 'wfo',
                                  'qsr', 'qns', 'qt', 'sfx', 'taum', 'windsp', 'precip', 'bosp_Qout',
                                  'bosp_Qin', 'mmean_S_total', 'bosp_S_in', 'CFL', 'NFL', 'CEM', 'NEM',
                                  'CDI', 'NDI', 'MIC', 'MES', 'BAC', 'DCL', 'DNL', 'DCS', 'DNS', 'NOS',
                                  'NHS', 'SIO', 'DIC', 'ODU', 'POC', 'PON', 'SID', 'AGG', 'GEL', 'NOC',
                                  'PHO', 'SMI', 'CHA', 'CHD', 'CHE', 'CHF', 'PAR', 'NPP', 'NPPint', 'Carbon_UptakeDiatoms2D',
                                  'Nitrogen_UptakeDiatoms2D', 'Carbon_UptakeEmiliana2D','Nitrogen_UptakeEmiliana2D', 'Carbon_UptakeFlagellates2D',
                                  'Nitrogen_UptakeFlagellates2D', 'shearrate', 'sinkingDIA', 'sinkingPOM', 'pH', 'pCO2', 'AirSeaDICFlux', 'TA']

        # Stores all the dataset names and location
        self.dataset_list = list()

        # Retreives the path to the folder containing the data (local or cluster)
        self.path_general = get_data_path(folder = folder)

        # Creation of the paths
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):

                # Checking month's validity
                if year == year_start and month < month_start or year == year_end and month_end < month:
                    continue

                # Updating to string for ease of use
                month_after         = str(month + 1) if 10 <= month + 1 else "0" + str(month + 1)
                month_after_after   = str(month + 2) if 10 <= month + 2 else "0" + str(month + 2)
                month_before        = str(month - 1) if 10 <= month - 1 else "0" + str(month - 1)
                month_before_before = str(month - 2) if 10 <= month - 2 else "0" + str(month - 2)
                month               = str(month) if 10 <= month else "0" + str(month)

                # Creating all list of possibilities for simulation name (check folder)
                years_sim = [f"BS_1d_19{80 + year    }0101_19{80 + year    }1231_",
                             f"BS_1d_19{80 + year - 1}1231_19{80 + year    }1231_",
                             f"BS_1d_19{80 + year    }0101_19{80 + year    }1230_",
                             f"BS_1d_19{80 + year    }0101_19{80 + year + 1}0101_"]

                months_sim = [f"198{year}{month}-198{year}{month    }",
                              f"198{year}{month}-198{year}{month_after}",
                              f"198{year}{month}-198{year}{month_after_after}",
                              f"198{year}{month_before}-198{year}{month}",
                              f"198{year}{month_before_before}-198{year}{month}"]

                extensions = [".nc4", ".nc"]

                # Used to determine if a path has been found or not
                path_found = False

                # Testing the possible paths
                for e in extensions:

                    # Getting out (1)
                    if path_found:
                        break

                    for m in months_sim:

                        # Getting out (2)
                        if path_found:
                            break

                        for y in years_sim:

                            # Creation of the tested path
                            current_path = f"{self.path_general}{y}{variable}_{m}{e}"

                            # Check if the path exists
                            if os.path.exists(current_path):
                                path_found = True
                                self.dataset_list.append(current_path)

                            # Getting out (3)
                            if path_found:
                                break

                # Something wrong, i.e. the title of the file was not found ! Need to update possibilities or maybe the simulation is not done yet
                if path_found == False:
                    print(f"ISSUE (Init) - Path not found for: Year ({year}), Month ({month}), Variable ({variable})")

        # Deleting duplicates (if month is 01_03, we will have it multiples times for months 1, 2, 3)
        self.dataset_list = list(dict.fromkeys(self.dataset_list))

        # Stores only the first dataset (for the sake of efficiency)
        self.data = xarray.open_dataset(self.dataset_list[0], engine = "h5netcdf", drop_variables = self.useless_variables)

        # Saving other relevant information
        self.year_start  = year_start
        self.year_end    = year_end
        self.month_start = month_start
        self.month_end   = month_end
        self.variable    = variable
        self.folder      = folder

    def translate(self, variable : str):
        r"""Used to translate the variable name to the one used in the dataset"""

        # Stores all the translations
        translations = {"temperature" : "votemper",
                        "salinity"    : "vosaline",
                        "oxygen"      : "DOX",
                        "chlorophyll" : "CHL",
                        "k_short"     : "KBIOS",
                        "k_long"      : "KBIOL"}

        return translations[variable]

    def get_bathymetry(self):
        r"""Used to retreive the bathymetry mask, i.e. the depth index at which we reach the bottom of the ocean (2D)"""

        # Path to the mesh file location
        path_mesh = get_mesh_path()

        # Loading the dataset containing information about the Black Sea mesh
        return xarray.open_dataset(path_mesh, engine = "h5netcdf").mbathy.data

    def get_mask(self, depth: int = None):
        r"""Used to retreive a mask of the Black Sea, i.e. 0 if land, 1 if the Black Sea. If depth is given, it will also set to 0 all regions below that depth"""

        # Path to the mesh file location
        path_mesh = get_mesh_path()

        # Loading the dataset containing information about the Black Sea mesh
        mesh_data = xarray.open_dataset(path_mesh, engine = "h5netcdf")

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

    def get_bottom(self, data: np.array, depth = None):
        r"""Used to retreive the data profile (2D) everywhere at the bottom of the Black Sea (None) of for all regions above a given depth"""

        # Security
        assert len(data.shape) == 4, f"ERROR (get_bottom), Incorrect data shape ({data.shape}), i.e. input dimensions should be (time, depth, y, x)"

        # Retreiving the bathymetry mask b(t, x, y) = z_bottom, i.e. index at which we found bottom of the sea
        bathy_mask = self.get_bathymetry()

        # Creation of x and y indexes to make manipulation
        x, y = np.arange(bathy_mask.shape[2]), np.arange(bathy_mask.shape[1])
        xidx = x.reshape(-1,1).repeat(len(y),axis=1).T
        yidx = y.reshape(-1,1).repeat(len(x),axis=1)

        # Retreiving the data everywhere at the bottom
        data = data[:, bathy_mask[0] - 1, yidx, xidx]

        # Retreiving for regions above depth treshold (product with 0, 1 mask applies the mask instantly) if needed
        return data if depth == None else data[:] * self.get_mask(depth = depth)

    def get_data(self, variable : str, type : str = "all", depth : int = None):
        r"""Used to retreive the data for a given variable and type (surface (2D), bottom (2D) or all (3D))"""

        # Security (1)
        assert variable in ["temperature", "salinity", "oxygen", "chlorophyll", "k_short", "k_long"], f"ERROR (get_data), Incorrect variable ({variable})"
        assert type in     ["surface", "bottom", "all"],                                              f"ERROR (get_data), Incorrect type ({type})"

        # Security (2)
        if variable in ["temperature", "salinity"]:
            assert self.variable == "grid_T", f"ERROR (get_data), Dataset is not grid_T ({self.variable})"
        else:
            assert self.variable == "ptrc_T", f"ERROR (get_data), Dataset is not ptrc_T ({self.variable})"

        # Security (3)
        if not depth == None:
            assert type in ["surface", "bottom"], f"ERROR (get_data), Incorrect type ({type}) if depth is given"

        # Translating the variable name
        variable = self.translate(variable)

        # Retreives the data for each day of the month at the surface
        dataset = self.data[variable][:, 0, :, :].data if type == "surface" else self.data[variable][:, :, :, :].data

        # Retreives the data in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):

            # Loading the new dataset
            data = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf", drop_variables = self.useless_variables)

            # Loading the field
            data = data[variable][:, 0, :, :].data if type == "surface" else data[variable][:, :, :, :].data

            # Concatenation of datasets along the time dimension
            dataset = np.concatenate((dataset, data), axis = 0)

        # Returns only the bottom values if needed
        return self.get_bottom(data = dataset, depth = depth) if type == "bottom" else dataset