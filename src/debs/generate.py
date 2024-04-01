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
# A simple script to preprocess the data (normalization, flipping, reshaping, ...) the data
#

import os
import json
import wandb
import xarray
import calendar
import numpy as np
from dataset import BlackSea_Dataset

if __name__ == "__main__":

  # Initialization
  months       = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  years        = [i for i in range(1980, 2023)]
  oxy_treshold = 63
  max_depth    = 200

  # Initial minimum and maximum values
  t_min, t_max           = 100, -100
  s_min, s_max           = 100, -100
  c_min, c_max           = 100, -100
  kshort_min, kshort_max = 100, -100
  klong_min, klong_max   = 100, -100
  o_min, o_max           = 100, -100

  # Information over the terminal (1)
  print("--------------------------------")
  print("Starting the data generation ...")
  print("--------------------------------")

  # Sending information to wandb
  wandb.init(project = "Generating data")

  # Finding the minimum and maximum values
  for y in years:
    for m in months:
      print(f"\n --------- Observing : {m}/{y} ---------")
      BSD_dataset = BlackSea_Dataset(year_start = y, year_end = y, month_start = m, month_end = m)

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
      t_min      = np.nanmin([t_min,      np.nanmin(data_temperature)])
      t_max      = np.nanmax([t_max,      np.nanmax(data_temperature)])
      s_min      = np.nanmin([s_min,      np.nanmin(data_salinity)])
      s_max      = np.nanmax([s_max,      np.nanmax(data_salinity)])
      c_min      = np.nanmin([c_min,      np.nanmin(data_chlorophyll)])
      c_max      = np.nanmax([c_max,      np.nanmax(data_chlorophyll)])
      kshort_min = np.nanmin([kshort_min, np.nanmin(data_kshort)])
      kshort_max = np.nanmax([kshort_max, np.nanmax(data_kshort)])
      klong_min  = np.nanmin([klong_min,  np.nanmin(data_klong)])
      klong_max  = np.nanmax([klong_max,  np.nanmax(data_klong)])
      o_min      = np.nanmin([o_min,      np.nanmin(data_oxygen)])
      o_max      = np.nanmax([o_max,      np.nanmax(data_oxygen)])

      # Displaying information over the terminal (2)
      print(f"Temperature:\nMin = {t_min}\nMax = {t_max}")
      print(f"Salinity:\nMin = {s_min}\nMax = {s_max}")
      print(f"Chlorophyll:\nMin = {c_min}\nMax = {c_max}")
      print(f"Kshort:\nMin = {kshort_min}\nMax = {kshort_max}")
      print(f"Klong:\nMin = {klong_min}\nMax = {klong_max}")
      print(f"Oxygen:\nMin = {o_min}\nMax = {o_max}")

      # Sending information to wandb
      wandb.log({"Year (Preprocessing)": y,
                 "Month (Preprocessing)": m,
                 "Temperature Min": t_min,
                 "Temperature Max": t_max,
                 "Salinity Min": s_min,
                 "Salinity Max": s_max,
                 "Chlorophyll Min": c_min,
                 "Chlorophyll Max": c_max,
                 "Kshort Min": kshort_min,
                 "Kshort Max": kshort_max,
                 "Klong Min": klong_min,
                 "Klong Max": klong_max,
                 "Oxygen Min": o_min,
                 "Oxygen Max": o_max})

  # Loading or computing fixed variables
  mask_BS     = BSD_dataset.get_mask(depth = None)
  mask_CS     = BSD_dataset.get_mask(depth = max_depth)
  bathy_METER = BSD_dataset.get_depth(unit = "meter")[0]
  bathy_INDEX = BSD_dataset.get_depth(unit = "index")[0]
  oxy_normal  = (oxy_treshold - o_min)/(o_max - o_min)

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

  # Information over the terminal (3)
  print("\n------------------------")
  print("Normalizing the data ...")
  print("------------------------")

  # Normalizing and saving the data
  for y in years:
    for m in months:
      print(f"\n --------- Processing : {m}/{y} ---------")
      BSD_dataset = BlackSea_Dataset(year_start = y, year_end = y, month_start = m, month_end = m)

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
      data_temperature = (data_temperature - t_min)      /(t_max      - t_min)
      data_salinity    = (data_salinity    - s_min)      /(s_max      - s_min)
      data_chlorophyll = (data_chlorophyll - c_min)      /(c_max      - c_min)
      data_kshort      = (data_kshort      - kshort_min) /(kshort_max - kshort_min)
      data_klong       = (data_klong       - klong_min)  /(klong_max  - klong_min)
      data_oxygen_CS  = (data_oxygen_CS    - o_min)      /(o_max      - o_min)
      data_oxygen_ALL = (data_oxygen_ALL   - o_min)      /(o_max      - o_min)

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
                      data  = oxy_normal,
                      attrs = {"Units": "-",
                              "Description": "Normalized Hypoxia Treshold (using the min and max values of the oxygen concentration)"})

      # Minimum and maximum values
      ds_oxy_min = xarray.DataArray(
                      name  = "OXYMIN",
                      data  = o_min,
                      attrs = {"Units": "mmol/m3",
                              "Description": "Minimum value of oxygen concentration at the bottom layer of the Black Sea"})

      ds_oxy_max = xarray.DataArray(
                      name  = "OXYMAX",
                      data  = o_max,
                      attrs = {"Units": "mmol/m3",
                              "Description": "Maximum value of oxygen concentration at the bottom layer of the Black Sea"})

      ds_temp_min = xarray.DataArray(
                      name  = "TEMPMIN",
                      data  = t_min,
                      attrs = {"Units": "degC",
                              "Description": "Minimum value of temperature at the surface layer of the Black Sea"})

      ds_temp_max = xarray.DataArray(
                      name  = "TEMPMAX",
                      data  = t_max,
                      attrs = {"Units": "degC",
                              "Description": "Maximum value of temperature at the surface layer of the Black Sea"})

      ds_sal_min = xarray.DataArray(
                      name  = "SALMIN",
                      data  = s_min,
                      attrs = {"Units": "1e-3",
                              "Description": "Minimum value of salinity at the surface layer of the Black Sea"})

      ds_sal_max = xarray.DataArray(
                      name  = "SALMAX",
                      data  = s_max,
                      attrs = {"Units": "1e-3",
                              "Description": "Maximum value of salinity at the surface layer of the Black Sea"})

      ds_chloro_min = xarray.DataArray(
                      name  = "CHLMIN",
                      data  = c_min,
                      attrs = {"Units": "mmol/m3",
                              "Description": "Minimum value of chlorophyll concentration at the surface layer of the Black Sea"})

      ds_chloro_max = xarray.DataArray(
                      name  = "CHLMAX",
                      data  = c_max,
                      attrs = {"Units": "mmol/m3",
                              "Description": "Maximum value of chlorophyll concentration at the surface layer of the Black Sea"})

      ds_kshort_min = xarray.DataArray(
                      name  = "KSHORTMIN",
                      data  = kshort_min,
                      attrs = {"Units": "-",
                              "Description": "Minimum value of reflectance (short wavelengths) at the surface layer of the Black Sea"})

      ds_kshort_max = xarray.DataArray(
                      name  = "KSHORTMAX",
                      data  = kshort_max,
                      attrs = {"Units": "-",
                              "Description": "Maximum value of reflectance (short wavelengths) at the surface layer of the Black Sea"})

      ds_klong_min = xarray.DataArray(
              name  = "KLONGMIN",
              data  = klong_min,
              attrs = {"Units": "-",
                      "Description": "Minimum value of reflectance (long wavelengths) at the surface layer of the Black Sea",})

      ds_klong_max = xarray.DataArray(
              name  = "KLONGMAX",
              data  = klong_max,
              attrs = {"Units": "-",
                      "Description": "Maximum value of reflectance (long wavelengths) at the surface layer of the Black Sea",})

      # Creation of the name
      file_name = f"BlackSea-DeepLearning_{y}_{m}.nc"

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
                              ds_temp_min,
                              ds_temp_max,
                              ds_sal_min,
                              ds_sal_max,
                              ds_chloro_min,
                              ds_chloro_max,
                              ds_kshort_min,
                              ds_kshort_max,
                              ds_klong_min,
                              ds_klong_max,
                              ds_oxy_min,
                              ds_oxy_max])

      # Adding metadata
      dataset.attrs = {"Author": "Victor Mangeleer",
                      "Contact": "vmangeleer@uliege.be",
                      "Description": "Preprocessed Black Sea Dataset (2D Formulation, surface to bottom) used for Deep Learning",
                      "Simulation Date": f"{y}-{m}"}

      # Saving the data
      dataset.to_netcdf(f"../../../../../../../scratch/acad/bsmfc/victor/data/{file_name}")

      # Displaing information over the terminal
      print(f"Data saved : {file_name}")

      # Sending information to wandb
      wandb.log({"Year (Processing)": y, "Month (Processing)": m})