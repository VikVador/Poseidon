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
# A class to handle easily datasets coming from the NEMO simulator
#
#
import os
import xarray
import numpy as np
import matplotlib.pyplot as plt


class BlackSea_Dataset():
    r"""A simple dataloader for black sea dataset"""
    
    def __init__(self, year_start: int, year_end: int, month_start: int, month_end: int, variable: str, folder:str = "output_HR004"):
        super().__init__()

        # Security (Level 1)
        assert year_start  in [i for i in range(10)], f"ERROR (Dataset, init) - Incorrect starting year ({year_start})"
        assert year_end    in [i for i in range(10)], f"ERROR (Dataset, init) - Incorrect ending year ({year_end})"
        assert month_start in [i for i in range(13)], f"ERROR (Dataset, init) - Incorrect starting month ({month_start})"
        assert month_start in [i for i in range(13)], f"ERROR (Dataset, init) - Incorrect ending month ({month_end})"
        assert year_start <= year_end,                f"ERROR (Dataset, init) - Incorrect years ({year_start} <= {year_end})"
        assert variable in ["grid_U", "grid_V", "grid_W", "grid_T", "ptrc_T", "btrc_T"], f"ERROR (Dataset, init) - Incorrect variable ({variable})"
        if year_start == year_end:
            assert month_start <= month_end, f" ERROR (Dataset, init) - Incorrect months ({month_start} <= {month_end})"
        
        # Stores all the dataset names and location
        self.dataset_list = list()

        # General path to each file
        self.path_general = f"../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS_BIO/{folder}/"
        
        # Creation of the paths
        for year in range(year_start, year_end + 1):
            for month in range(1, 13):
    
                # Cheking month's validity
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

                # Something wrong, i.e. the title of the file was not found ! Need to update possibilities
                if path_found == False:
                    print(f"ISSUE (Init) - Path not found for: Year ({year}), Month ({month}), Variable ({variable})")

        # Deleting duplicates (if month is 01_03, we will have it multiples times for months 1, 2, 3)
        self.dataset_list = list(dict.fromkeys(self.dataset_list))

        # Stores only the first dataset (for the sake of efficiency)
        self.data = xarray.open_dataset(self.dataset_list[0], engine = "h5netcdf")

        # Saving information
        self.year_start = year_start
        self.year_end = year_end
        self.month_start = month_start
        self.month_end = month_end
        self.variable = variable
        self.folder = folder

    def get_bathymetry(self, to_np_array: bool = True):
        r"""Used to retreive the bathymetry mask, i.e. the depth index at which we reach the bottom of the ocean (2D)"""
        
        # Path to the file location
        path_mesh = "../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS/mesh_mask.nc_new59_CMCC_noAzov"

        # Loading the dataset containing information about the Black Sea mesh
        mesh_data = xarray.open_dataset(path_mesh, engine = "h5netcdf")
        
        return mesh_data.mbathy.data if to_np_array else mesh_data.mbathy

    def get_blacksea_mask(self, to_np_array: bool = True, depth: int = None):
        r"""Used to retreive the black sea mask, i.e. a mask where 0 = the depth is below treshold and 1 = above treshold"""
        
        # Path to the file location
        path_mesh = "../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS/mesh_mask.nc_new59_CMCC_noAzov"

        # Loading the dataset containing information about the Black Sea mesh
        mesh_data = xarray.open_dataset(path_mesh, engine = "h5netcdf")

        # Loading the full Black sea mask (0 if land, 1 if the Black Sea)
        bs_mask = mesh_data.tmask[0, 0].data
        
        # Checks if we want to retreive a specific part of the Black Sea, i.e. continental shelf found for all depth > 120m
        if not depth == None:
            
            # Retreives the bottom depth in [m] for each pixel
            depth_values = mesh_data.bathy_metry.data[0]

            # Remove all information for regions located below the given depth
            bs_mask[depth <= depth_values] = 0

            # Returning the new mask (processed)
            return bs_mask

        return bs_mask if to_np_array else mesh_data.tmask[0, 0]
    
    def get_temperature(self, to_np_array: bool = True):
        r"""Used to retreive the surface temperature (2D)"""
        
        # Security        
        assert self.variable == "grid_T", f"ERROR (get_temperature), Dataset is not grid_T ({self.variable})"
        
        # Retreives the temperature for each day of the month at the surface
        data_temperature = self.data.votemper[:,0,:,:].data if to_np_array else self.data.votemper[:,0,:,:]

        # Retreives the temperature in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the temperature field
            dataset = dataset.votemper[:,0,:,:].data if to_np_array else dataset.votemper[:,0,:,:]

            # Concatenation of datasets along the time dimension
            data_temperature = np.concatenate((data_temperature, dataset), axis = 0) if to_np_array else \
                                xarray.concat([data_temperature, dataset], dim  = "time_counter")
            
        return data_temperature

    def get_salinity(self, to_np_array: bool = True):
        r"""Used to retreive the surface salinity (2D)"""

        # Security        
        assert self.variable == "grid_T", f"ERROR (get_salinity), Dataset is not grid_T ({self.variable})"
        
        # Retreives the salinity for each day of the month at the surface
        data_salinity = self.data.vosaline[:,0,:,:].data if to_np_array else self.data.vosaline[:,0,:,:]

        # Retreives the salinity in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the salinity field
            dataset = dataset.vosaline[:,0,:,:].data if to_np_array else dataset.vosaline[:,0,:,:]

            # Concatenation of datasets along the time dimension
            data_salinity = np.concatenate((data_salinity, dataset), axis = 0) if to_np_array else \
                             xarray.concat([data_salinity, dataset], dim  = "time_counter")
            
        return data_salinity
    
    def get_oxygen(self, to_np_array: bool = True):
        r"""Used to retreive the full oxygen profile (3D)"""
        
        # Security        
        assert self.variable == "ptrc_T", f"ERROR (get_oxygen), Dataset is not ptrc_T ({self.variable})"

        # Retreives the oxygen for each day of the month in the whole sea
        data_oxygen = self.data.DOX.data if to_np_array else self.data.DOX

        # Retreives the oxygen in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the oxygen field
            dataset = dataset.DOX.data if to_np_array else dataset.DOX

            # Concatenation of datasets along the time dimension
            data_oxygen = np.concatenate((data_oxygen, dataset), axis = 0) if to_np_array else \
                           xarray.concat([data_oxygen, dataset], dim  = "time_counter")

        return data_oxygen
        
    def get_oxygen_bottom(self, depth = None):
        r"""Used to retreive the oxygen profile (2D), i.e. the concentration everywhere (None) or for all regions above a given depth"""
        
        # Security        
        assert self.variable == "ptrc_T", f"ERROR (get_oxygen), Dataset is not ptrc_T ({self.variable})"

        # Retreiving the bathymetry mask b(t, x, y) = z_bottom, i.e. index at which we found bottom of the sea
        bathy_mask = self.get_bathymetry()

        # Retreiving oxygen levels in the whole sea
        data_oxygen = self.get_oxygen()

        # Creation of x and y indexes to make manipulation, i.e. ox_bottom = ox(t, b(x, y), y, x)
        x, y = np.arange(bathy_mask.shape[2]), np.arange(bathy_mask.shape[1])
        xidx = x.reshape(-1,1).repeat(len(y),axis=1).T
        yidx = y.reshape(-1,1).repeat(len(x),axis=1)

        # Retreiving the oxygen concentrations everywhere
        data_oxygen = data_oxygen[:, bathy_mask[0] - 1, yidx, xidx]

        # Retreiving oxygen concentration everywhere or for regions above depth treshold (product with 0, 1 mask applies the mask instantly)
        data_oxygen = data_oxygen if depth == None else data_oxygen[:] * self.get_blacksea_mask(depth = depth)

        # A bit of post processing, i.e. setting NANs and < 0 concentrations (not physical) to 0.
        return np.clip(np.nan_to_num(data_oxygen, nan = 0.), 0, None)

    def get_chlorophyll(self, to_np_array: bool = True):
        r"""Used to retreive the surface chlorophyll (2D)"""
        
        # Security        
        assert self.variable == "ptrc_T", f"ERROR (get_chlorophyll), Dataset is not ptrc_T ({self.variable})"

        # Retreives the chlorophyll for each day of the month at the surface
        data_chlorophyll = self.data.CHL[:,0,:,:].data if to_np_array else self.data.CHL[:,0,:,:]

        # Retreives the chlorophyll in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the chlorophyll field
            dataset = dataset.CHL[:,0,:,:].data if to_np_array else dataset.CHL[:,0,:,:]

            # Concatenation of datasets along the time dimension
            data_chlorophyll = np.concatenate((data_chlorophyll, dataset), axis = 0) if to_np_array else \
                                xarray.concat([data_chlorophyll, dataset], dim  = "time_counter")
            
        return data_chlorophyll
    
    def get_light_attenuation_coefficient_short_waves(self, to_np_array: bool = True):
        r"""Used to retreive the surface light attenuation coefficient (k) for short waves number (2D)"""
        
        # Security        
        assert self.variable == "ptrc_T", f"ERROR (get_light_attenuation_coefficient_short_waves), Dataset is not ptrc_T ({self.variable})"

        # Retreives the k_short coefficient for each day of the month at the surface
        data_k_short = self.data.KBIOS[:,0,:,:].data if to_np_array else self.data.KBIOS[:,0,:,:]

        # Retreives the k_short in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the k_short
            dataset = dataset.KBIOS[:,0,:,:].data if to_np_array else dataset.KBIOS[:,0,:,:]

            # Concatenation of datasets along the time dimension
            data_k_short = np.concatenate((data_k_short, dataset), axis = 0) if to_np_array else \
                            xarray.concat([data_k_short, dataset], dim  = "time_counter")
            
        return data_k_short
        
    def get_light_attenuation_coefficient_long_waves(self, to_np_array: bool = True):
        r"""Used to retreive the surface light attenuation coefficient (k) for long waves number (2D)"""
        
        # Security        
        assert self.variable == "ptrc_T", f"ERROR (get_light_attenuation_coefficient_long_waves), Dataset is not ptrc_T ({self.variable})"

        #Retreives the k_long coefficient for each day of the month at the surface
        data_k_long = self.data.KBIOL[:,0,:,:].data if to_np_array else self.data.KBIOL[:,0,:,:]

        # Retreives the k_long in the other datasets and concatenates
        for d in range(1, len(self.dataset_list)):
            
            # Loading the new dataset
            dataset = xarray.open_dataset(self.dataset_list[d], engine = "h5netcdf")

            # Loading the k_long
            dataset = dataset.KBIOL[:,0,:,:].data if to_np_array else dataset.KBIOL[:,0,:,:]

            # Concatenation of datasets along the time dimension
            data_k_long = np.concatenate((data_k_long, dataset), axis = 0) if to_np_array else \
                           xarray.concat([data_k_long, dataset], dim  = "time_counter")
            
        return data_k_long
        