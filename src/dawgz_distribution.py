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
# A DAWGZ script to generate jobs
#
#
import os
import sys
import xarray
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dataset      import BlackSea_Dataset
from tools        import BlackSea_Tools
from distribution import BlackSea_Distributions
from dawgz        import job, after, ensure, schedule

# -----------------
#   Combinations
# -----------------
# 1) Distributions over the 10 years
month_start = [1]
month_end   = [12]
year_start  = [1]
year_end    = [9]

# 2) Distributions over the years
for i in range(1, 9):
    month_start.append(1)
    month_end.append(12)
    year_start.append(i)
    year_end.append(i + 1)

# 3) Distribution over the months
for i in range(1, 9):
    for j in range(1, 12):
        month_start.append(j)
        month_end.append(j + 1)
        year_start.append(i)
        year_end.append(i)

# Total number of jobs
nb_tasks = len(month_start)

# ----------------------
#   Scheduler function
# ----------------------
@job(array=nb_tasks, cpus=1, ram='64GB', time='12:00:00', project='bsmfc', user='vmangeleer@uliege.be', type='FAIL')
def compute_distribution(i: int):

    # Retreiving corresponding time period
    start_month, end_month, start_year, end_year = month_start[i],  month_end[i],  year_start[i],  year_end[i]

    # Date extension for result file
    date = f"8{start_year}-{start_month}_to_8{end_year}-{end_month}"

    # Information over terminal (1)
    print("Date: {date} - Loading the data")

    # Intialization of the dataset handler !
    Dataset_physical = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "grid_T")
    Dataset_bio      = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "ptrc_T")

    # Loading the different field values
    data_temperature   = Dataset_physical.get_temperature()
    data_salinity      = Dataset_physical.get_salinity()
    data_oxygen_bottom = Dataset_bio.get_oxygen_bottom()
    data_chlorophyll   = Dataset_bio.get_chlorophyll()
    data_kshort        = Dataset_bio.get_light_attenuation_coefficient_short_waves()
    data_klong         = Dataset_bio.get_light_attenuation_coefficient_long_waves()

    # Creation of a list containing all the datasets whose distribution must be analyze
    datasets = [data_oxygen_bottom, data_temperature, data_salinity, data_chlorophyll, data_klong, data_kshort]

    # ------- Extracting distributions from the data -------
    #
    # Information over terminal (2)
    print("Loading the distributions handler")

    # Loading distribution handler tool
    distribution_handler = BlackSea_Distributions(subpopulation_percentage = 50, dataloader = Dataset_physical, datasets = datasets,
                                                  year_start = start_year, year_end = end_year, month_start = start_month, month_end = end_month)

    # Information over terminal (3)
    print("Loading the marginals")

    # Computing marginal distributions
    distribution_handler.plot_marginal(save = True, file_name = f"../analysis/__distributions__/marginal/marginal_{date}.png")

    # Information over terminal (4)
    print("Loading the joints")

    # Computing joint distributions
    distribution_handler.plot_joint(save = True, file_name = f"../analysis/__distributions__/joint/joint_{date}.png")

if __name__ == "__main__":
    schedule(compute_distribution, name='distributions', backend='slurm', export='ALL')
