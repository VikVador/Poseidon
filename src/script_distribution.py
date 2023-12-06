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
# A simple script to analyze data (Focus towards distributions)
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
from matplotlib.animation import FuncAnimation

from dataset      import BlackSea_Dataset
from tools        import BlackSea_Tools
from distribution import BlackSea_Distributions
from dawgz import job, after, ensure, schedule

if __name__ == "__main__":

    # ------- Parsing the command-line arguments -------
    #
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE:      python script.py  --start_year    <X>
                                  --end_year      <X>
                                  --start_month   <X>
                                  --end_month     <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--start_year',
        help  = 'Starting year to collect data',
        type  = int)

    parser.add_argument(
        '--end_year',
        help = 'Ending year to collect data',
        type = int)

    parser.add_argument(
        '--start_month',
        help = 'Starting month to collect data',
        type = int)

    parser.add_argument(
        '--end_month',
        help = 'Ending month to collect data',
        type = int)

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Placing into correct variables for the ease of use
    start_year  = args.start_year
    end_year    = args.end_year
    start_month = args.start_month
    end_month   = args.end_month
    
    # ------- Loading the data -------
    #
    # Information over terminal (1)
    print("Loading the data")
    
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

    # Date extension for result file
    date = f"{start_month}-8{start_year}_to_{end_month}-8{end_year}"
    
    # ------- Extracting distributions from the data -------
    #
    # Information over terminal (2)
    print("Loading the distributions handler")

    # Loading distribution handler tool
    distribution_handler = BlackSea_Distributions(subpopulation_percentage = 1, dataloader = Dataset_physical, datasets = datasets, 
                                                  year_start = start_year, year_end = end_year, month_start = start_month, month_end = end_month)

    # Information over terminal (3)
    print("Loading the marginals")
    
    # Computing marginal distributions
    distribution_handler.plot_marginal(save = True, file_name = f"../analysis/__distributions__/marginal/marginal_{date}.png")

    # Information over terminal (4)
    print("Loading the joints")
    
    # Computing joint distributions
    distribution_handler.plot_joint(save = True, file_name = f"../analysis/__distributions__/joint/joint_{date}.png")

    


