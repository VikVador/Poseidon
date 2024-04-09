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

# Maths
# -----
#
# Number of timesteps accross all the years : (15706, 256, 576)
# Number of valid pixels (mask applied)     : (15706, 74068)
# In total                                  :  15706 * 74068 = 1163312008

# Initialization
months       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years        = [i for i in range(2002, 2023)]
oxy_treshold = 63
max_depth    = 200

# Initial mean and variance values
t_mean,           t_variance = 15.634022032720223 , 47.84012854142219
s_mean,           s_variance = 17.786955222420445 , 0.9846352557378572
c_mean,           c_variance = 1.3657086471744726 , 8.609137194548328
kshort_mean, kshort_variance = 0.13270733572890703, 0.015299949430704457
klong_mean,   klong_variance = 0.3810429258351644 , 0.008328035745959672
o_mean,           o_variance = 58.30502765686229  , 12472.950808966465

# Loading the mask to make sure we do not take into account values outside the Black Sea
tool            = BlackSea_Dataset_UNPROCESSED(year_start = 1980, year_end = 1980, month_start = 1, month_end = 1)
processing_mask = tool.get_mask(depth = None)

# Total number of elements
N_elements = 1163312008

# Information over the terminal (1)
print("---------------------------------")
print("Starting to look for the mean ...")
print("---------------------------------")

# Sending information to wandb
wandb.init(project = "Generating data (Standardized)")

# Finding the average values
for y in years:
        for m in months:
                print(f"\n --------- Observing : {m}/{y} ---------")
                BSD_dataset = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

                # Loading the data
                data_temperature   = BSD_dataset.get_data(variable = "temperature", region = "surface")
                data_salinity      = BSD_dataset.get_data(variable = "salinity",    region = "surface")
                data_chlorophyll   = BSD_dataset.get_data(variable = "chlorophyll", region = "surface")
                data_kshort        = BSD_dataset.get_data(variable = "kshort",      region = "surface")
                data_klong         = BSD_dataset.get_data(variable = "klong",       region = "surface")
                data_oxygen        = BSD_dataset.get_data(variable = "oxygen",      region = "bottom")

                # Cliping negative values (needed for unphysical results due to numerical schemes)
                data_oxygen = np.clip(data_oxygen, 0, None)

                # Finding the minimum and maximum values
                t_mean      += np.nansum(data_temperature[:, processing_mask == 1])/N_elements
                s_mean      += np.nansum(data_salinity[:,    processing_mask == 1])/N_elements
                c_mean      += np.nansum(data_chlorophyll[:, processing_mask == 1])/N_elements
                kshort_mean += np.nansum(data_kshort[:,      processing_mask == 1])/N_elements
                klong_mean  += np.nansum(data_klong[:,       processing_mask == 1])/N_elements
                o_mean      += np.nansum(data_oxygen[:,      processing_mask == 1])/N_elements

                # Displaying information over the terminal (2)
                print(f"Temperature:\n Mean = {t_mean}")
                print(f"Salinity:\n Mean = {s_mean}")
                print(f"Chlorophyll:\n Mean = {c_mean}")
                print(f"Kshort:\n Mean = {kshort_mean}")
                print(f"Klong:\n Mean = {klong_mean}")
                print(f"Oxygen:\n Mean = {o_mean}")

                # Sending information to wandb
                wandb.log({"Year (Preprocessing - Mean)"  : y,
                           "Month (Preprocessing - Mean)" : m,
                           "Temperature Mean"             : t_mean,
                           "Salinity Mean"                : s_mean,
                           "Chlorophyll Mean"             : c_mean,
                           "Kshort Mean"                  : kshort_mean,
                           "Klong Mean"                   : klong_mean,
                           "Oxygen Mean"                  : o_mean})

# Information over the terminal (3)
print("-------------------------------------")
print("Starting to look for the variance ...")
print("-------------------------------------")

# Finding the variance values
for y in years:
        for m in months:
                print(f"\n --------- Observing : {m}/{y} ---------")
                BSD_dataset = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

                # Loading the data
                data_temperature   = BSD_dataset.get_data(variable = "temperature", region = "surface")
                data_salinity      = BSD_dataset.get_data(variable = "salinity",    region = "surface")
                data_chlorophyll   = BSD_dataset.get_data(variable = "chlorophyll", region = "surface")
                data_kshort        = BSD_dataset.get_data(variable = "kshort",      region = "surface")
                data_klong         = BSD_dataset.get_data(variable = "klong",       region = "surface")
                data_oxygen        = BSD_dataset.get_data(variable = "oxygen",      region = "bottom")

                # Cliping negative values (needed for unphysical results due to numerical schemes)
                data_oxygen = np.clip(data_oxygen, 0, None)

                # Finding the minimum and maximum values
                t_variance  += np.nansum((data_temperature[:, processing_mask == 1] - t_mean)      ** 2)/N_elements
                s_variance  += np.nansum((data_salinity[:,    processing_mask == 1] - s_mean)      ** 2)/N_elements
                c_variance  += np.nansum((data_chlorophyll[:, processing_mask == 1] - c_mean)      ** 2)/N_elements
                kshort_variance += np.nansum((data_kshort[:,  processing_mask == 1] - kshort_mean) ** 2)/N_elements
                klong_variance  += np.nansum((data_klong[:,   processing_mask == 1] - klong_mean)  ** 2)/N_elements
                o_variance  += np.nansum((data_oxygen[:,      processing_mask == 1] - o_mean)      ** 2)/N_elements

                # Displaying information over the terminal (4)
                print(f"Temperature:\n Variance = {t_variance}")
                print(f"Salinity:\n Variance = {s_variance}")
                print(f"Chlorophyll:\n Variance = {c_variance}")
                print(f"Kshort:\n Variance = {kshort_variance}")
                print(f"Klong:\n Variance = {klong_variance}")
                print(f"Oxygen:\n Variance = {o_variance}")

                # Sending information to wandb
                wandb.log({"Year (Preprocessing - Variance)"  : y,
                           "Month (Preprocessing - Variance)" : m,
                           "Temperature Variance"             : t_variance,
                           "Salinity Variance"                : s_variance,
                           "Chlorophyll Variance"             : c_variance,
                           "Kshort Variance"                  : kshort_variance,
                           "Klong Variance"                   : klong_variance,
                           "Oxygen Variance"                  : o_variance})

# Computing the standard deviation
t_std      = np.sqrt(t_variance)
s_std      = np.sqrt(s_variance)
c_std      = np.sqrt(c_variance)
kshort_std = np.sqrt(kshort_variance)
klong_std  = np.sqrt(klong_variance)
o_std      = np.sqrt(o_variance)

# Sending information to wandb
wandb.log({"Temperature Standard Deviation" : t_std,
           "Salinity Standard Deviation"    : s_std,
           "Chlorophyll Standard Deviation" : c_std,
           "Kshort Standard Deviation"      : kshort_std,
           "Klong Standard Deviation"       : klong_std,
           "Oxygen Standard Deviation"      : o_std})

# Loading or computing fixed variables
mask_BS     = BSD_dataset.get_mask(depth = None)
mask_CS     = BSD_dataset.get_mask(depth = max_depth)
bathy_METER = BSD_dataset.get_depth(unit = "meter")[0]
bathy_INDEX = BSD_dataset.get_depth(unit = "index")[0]
oxy_standa  = (oxy_treshold - o_mean)/o_std

# Flipping vertically the data
mask_BS     = np.flip(mask_BS,     axis = 0)
mask_CS     = np.flip(mask_CS,     axis = 0)
bathy_METER = np.flip(bathy_METER, axis = 0)
bathy_INDEX = np.flip(bathy_INDEX, axis = 0)

# Removing dimensions to be a multiple of 2
mask_BS     = mask_BS[2:, 2:]
mask_CS     = mask_CS[2:, 2:]
bathy_METER = bathy_METER[2:, 2:]
bathy_INDEX = bathy_INDEX[2:, 2:]


# Information over the terminal (5)
print("\n--------------------------")
print("Standardizing the data ...")
print("--------------------------")

# Normalizing and saving the data
for y in years:
        for m in months:
                print(f"\n --------- Processing : {m}/{y} ---------")
                BSD_dataset = BlackSea_Dataset_UNPROCESSED(year_start = y, year_end = y, month_start = m, month_end = m)

                # Loading the different inputs
                data_temperature   = BSD_dataset.get_data(variable = "temperature", region = "surface")
                data_salinity      = BSD_dataset.get_data(variable = "salinity",    region = "surface")
                data_chlorophyll   = BSD_dataset.get_data(variable = "chlorophyll", region = "surface")
                data_kshort        = BSD_dataset.get_data(variable = "kshort",      region = "surface")
                data_klong         = BSD_dataset.get_data(variable = "klong",       region = "surface")
                data_oxygen_CS     = BSD_dataset.get_data(variable = "oxygen",      region = "bottom", depth = max_depth)
                data_oxygen_ALL    = BSD_dataset.get_data(variable = "oxygen",      region = "bottom", depth = None)

                # Cliping negative values
                data_oxygen_CS  = np.clip(data_oxygen_CS,  0, None)
                data_oxygen_ALL = np.clip(data_oxygen_ALL, 0, None)

                # Normalizing the data
                data_temperature = (data_temperature  - t_mean)      /t_std
                data_salinity    = (data_salinity     - s_mean)      /s_std
                data_chlorophyll = (data_chlorophyll  - c_mean)      /c_std
                data_kshort      = (data_kshort       - kshort_mean) /kshort_std
                data_klong       = (data_klong        - klong_mean)  /klong_std
                data_oxygen_CS   = (data_oxygen_CS    - o_mean)      /o_std
                data_oxygen_ALL  = (data_oxygen_ALL   - o_mean)      /o_std

                # Flipping vertically the data (Black Sea is upside down)
                data_temperature = np.flip(data_temperature, axis = 1)
                data_salinity    = np.flip(data_salinity,    axis = 1)
                data_chlorophyll = np.flip(data_chlorophyll, axis = 1)
                data_kshort      = np.flip(data_kshort,      axis = 1)
                data_klong       = np.flip(data_klong,       axis = 1)
                data_oxygen_CS   = np.flip(data_oxygen_CS,  axis = 1)
                data_oxygen_ALL  = np.flip(data_oxygen_ALL, axis = 1)

                # Removing dimensions to be a multiple of 2
                data_temperature = data_temperature[:, 2:, 2:]
                data_salinity    = data_salinity[:,    2:, 2:]
                data_chlorophyll = data_chlorophyll[:, 2:, 2:]
                data_kshort      = data_kshort[:,      2:, 2:]
                data_klong       = data_klong[:,       2:, 2:]
                data_oxygen_CS  = data_oxygen_CS[:,    2:, 2:]
                data_oxygen_ALL = data_oxygen_ALL[:,   2:, 2:]

                # Hiding the land
                data_temperature[:, mask_BS == 0] = np.nan
                data_salinity[   :, mask_BS == 0] = np.nan
                data_chlorophyll[:, mask_BS == 0] = np.nan
                data_kshort[     :, mask_BS == 0] = np.nan
                data_klong[      :, mask_BS == 0] = np.nan
                data_oxygen_CS[  :, mask_BS == 0] = np.nan
                data_oxygen_ALL[ :, mask_BS == 0] = np.nan

                # Creation of the different xarrays
                #
                # Physical and Biogeochemical variables
                ds_oxy_ALL = xarray.DataArray(
                                name  = "OXY",
                                data  = data_oxygen_ALL,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Oxygen concentration at the bottom layer of the Black Sea"})

                ds_oxy_CS = xarray.DataArray(
                                name  = "OXYCS",
                                data  = data_oxygen_CS,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Oxygen concentration at the bottom layer of the Black Sea continental shelf (< 200m)"})

                ds_temp = xarray.DataArray(
                                name  = "TEMP",
                                data  = data_temperature,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "degC",
                                        "Description": "Temperature at the surface layer of the Black Sea"})

                ds_sal = xarray.DataArray(
                                name  = "SAL",
                                data  = data_salinity,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "1e-3",
                                        "Description": "Salinity at the surface layer of the Black Sea"})

                ds_chloro = xarray.DataArray(
                                name  = "CHL",
                                data  = data_chlorophyll,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Chlorophyll concentration at the surface layer of the Black Sea"})

                ds_kshort = xarray.DataArray(
                                name  = "KSHORT",
                                data  = data_kshort,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "-",
                                        "Description": "Reflectance (short wavelenghts) at the surface layer of the Black Sea"})

                ds_klong = xarray.DataArray(
                                name  = "KLONG",
                                data  = data_klong,
                                dims  = ["time", "x", "y"],
                                attrs = {"Units": "-",
                                        "Description": "Reflectance (long wavelenghts) at the surface layer of the Black Sea"})

                # Masks and bathymetry
                ds_mask_BS = xarray.DataArray(
                                name  = "MASK",
                                data  = mask_BS,
                                dims  = ["x", "y"],
                                attrs = {"Description": "Mask of the Black Sea"})

                ds_mask_CS = xarray.DataArray(
                                name  = "MASKCS",
                                data  = mask_CS,
                                dims  = ["x", "y"],
                                attrs = {"Description": "Mask of the Black Sea highlighting the continental shelf (< 200m)"})

                ds_bathy_M = xarray.DataArray(
                                name  = "BATHYM",
                                data  = bathy_METER,
                                dims  = ["x", "y"],
                                attrs = {"Description": "Bathymetry in meters, i.e. maximum depth of a region"})

                ds_bathy_I = xarray.DataArray(
                                name  = "BATHYI",
                                data  = bathy_INDEX,
                                dims  = ["x", "y"],
                                attrs = {"Description": "Bathymetry indices, i.e. indexes at which we found the bottom value in the 59 vertical layers"})

                ds_tresh_real = xarray.DataArray(
                                name  = "HYPO",
                                data  = oxy_treshold,
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Hypoxia Treshold"})

                ds_tresh_norm = xarray.DataArray(
                                name  = "HYPON",
                                data  = oxy_standa,
                                attrs = {"Units": "-",
                                        "Description": "Standardized Hypoxia Treshold (using the mean and variance values of the oxygen concentration)"})

                # Minimum and maximum values
                ds_oxy_mean = xarray.DataArray(
                                name  = "OXYMEAN",
                                data  = o_mean,
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Mean value of oxygen concentration at the bottom layer of the Black Sea"})

                ds_oxy_variance = xarray.DataArray(
                                name  = "OXYVAR",
                                data  = o_variance,
                                attrs = {"Units": "[mmol/m3]^2",
                                        "Description": "Variance value of oxygen concentration at the bottom layer of the Black Sea"})

                ds_temp_mean = xarray.DataArray(
                                name  = "TEMPMEAN",
                                data  = t_mean,
                                attrs = {"Units": "degC",
                                        "Description": "Mean value of temperature at the surface layer of the Black Sea"})

                ds_temp_variance = xarray.DataArray(
                                name  = "TEMPVAR",
                                data  = t_variance,
                                attrs = {"Units": "[degC]^2",
                                        "Description": "Variance value of temperature at the surface layer of the Black Sea"})

                ds_sal_mean = xarray.DataArray(
                                name  = "SALMEAN",
                                data  = s_mean,
                                attrs = {"Units": "1e-3",
                                        "Description": "Mean value of salinity at the surface layer of the Black Sea"})

                ds_sal_variance = xarray.DataArray(
                                name  = "SALVAR",
                                data  = s_variance,
                                attrs = {"Units": "[1e-3]^2",
                                        "Description": "Variance value of salinity at the surface layer of the Black Sea"})

                ds_chloro_mean = xarray.DataArray(
                                name  = "CHLMEAN",
                                data  = c_mean,
                                attrs = {"Units": "mmol/m3",
                                        "Description": "Mean value of chlorophyll concentration at the surface layer of the Black Sea"})

                ds_chloro_variance = xarray.DataArray(
                                name  = "CHLVAR",
                                data  = c_variance,
                                attrs = {"Units": "[mmol/m3]^2",
                                        "Description": "Variance value of chlorophyll concentration at the surface layer of the Black Sea"})

                ds_kshort_mean = xarray.DataArray(
                                name  = "KSHORTMEAN",
                                data  = kshort_mean,
                                attrs = {"Units": "-",
                                        "Description": "Mean value of reflectance (short wavelengths) at the surface layer of the Black Sea"})

                ds_kshort_variance = xarray.DataArray(
                                name  = "KSHORTVAR",
                                data  = kshort_variance,
                                attrs = {"Units": "-",
                                        "Description": "Variance value of reflectance (short wavelengths) at the surface layer of the Black Sea"})

                ds_klong_mean = xarray.DataArray(
                                name  = "KLONGMEAN",
                                data  = klong_mean,
                                attrs = {"Units": "-",
                                        "Description": "Mean value of reflectance (long wavelengths) at the surface layer of the Black Sea",})

                ds_klong_variance = xarray.DataArray(
                                name  = "KLONGVAR",
                                data  = klong_variance,
                                attrs = {"Units": "-",
                                        "Description": "Variance value of reflectance (long wavelengths) at the surface layer of the Black Sea",})

                # Creation of the name
                file_name = f"BlackSea-DeepLearning_Standardized_{y}_{m}.nc"

                # Concatenating everything
                dataset = xarray.merge([ds_temp,
                                        ds_sal,
                                        ds_chloro,
                                        ds_kshort,
                                        ds_klong,
                                        ds_oxy_ALL,
                                        ds_oxy_CS,
                                        ds_mask_BS,
                                        ds_mask_CS,
                                        ds_bathy_M,
                                        ds_bathy_I,
                                        ds_tresh_real,
                                        ds_tresh_norm,
                                        ds_temp_mean,
                                        ds_temp_variance,
                                        ds_sal_mean,
                                        ds_sal_variance,
                                        ds_chloro_mean,
                                        ds_chloro_variance,
                                        ds_kshort_mean,
                                        ds_kshort_variance,
                                        ds_klong_mean,
                                        ds_klong_variance,
                                        ds_oxy_mean,
                                        ds_oxy_variance])

                # Adding metadata
                dataset.attrs = {"Author": "Victor Mangeleer",
                                "Contact": "vmangeleer@uliege.be",
                                "Description": "Preprocessed (Standardized) Black Sea Dataset (2D Formulation, surface to bottom) used for Deep Learning",
                                "Simulation Date": f"{y}-{m}"}

                # Saving the data
                dataset.to_netcdf(f"../../../../../../../scratch/acad/bsmfc/victor/data/standardized/{file_name}")

                # Displaing information over the terminal
                print(f"Data saved : {file_name}")

                # Sending information to wandb
                wandb.log({"Year (Processing)": y, "Month (Processing)": m})