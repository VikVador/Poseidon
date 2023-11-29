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
# A simple script to generate data !
#
#
import os
import sys
import xarray
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from dataset import BlackSea_Dataset
from tools   import BlackSea_Tools

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
    
    # Security
    assert start_year <= end_year, f"ERROR (script.py) - The starting year must be <= to the end year ({start_year} <= {end_year})"
    if start_year == end_year:
        assert start_month < end_month, f"ERROR (script.py) - The starting month must be <= to the end onth ({start_month} <= {end_month})"

    
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
    
    # ------- Extracting information from the data -------
    #
    # Information over terminal (2)
    print("Loading the tools")
    
    # Loading the dataset tool !
    tool_temperature = BlackSea_Tools(Dataset_physical, data_temperature)
    tool_salinity    = BlackSea_Tools(Dataset_physical, data_salinity)
    tool_oxygen      = BlackSea_Tools(Dataset_bio,      data_oxygen_bottom)
    tool_chlorophyll = BlackSea_Tools(Dataset_bio,      data_chlorophyll)
    tool_kshort      = BlackSea_Tools(Dataset_bio,      data_kshort)
    tool_klong       = BlackSea_Tools(Dataset_bio,      data_klong)
    
    # Date extension for result file
    date = f"{start_month}-8{start_year}_to_{end_month}-8{end_year}"
    
    # ------- Evolution plots -------
    #
    # Information over terminal (3)
    print("Evolution plots")

    tool_temperature.plot_line("Temperature [C°]" ,     save = True, file_name = f"temperature_{date}")
    tool_salinity.plot_line(      "Salinity [ppt]",     save = True, file_name = f"salinity_{date}")
    tool_oxygen.plot_line(          "Oxygen [mmol/m3]", save = True, file_name = f"oxygen_{date}")
    tool_chlorophyll.plot_line("Chlorophyll [mmol/m3]", save = True, file_name = f"chlorophyll_{date}")
    tool_klong.plot_line(           "K-Long [-]",       save = True, file_name = f"klong_{date}")
    tool_kshort.plot_line(         "K-Short [-]",       save = True, file_name = f"kshort_{date}")

    # ------- Mask plots -------
    #
    # Information over terminal (4)
    print("Mask plot")
    
    tool_oxygen.plot_treshold(save = True, file_name = f"oxygen_ratios_{date}")

    # ------- Animation plots -------
    #
    # Information over terminal 5()
    print("Aninmation")
    
    tool_temperature.plot_animation(f"../images/temperature_animation_{date}.gif", ylabel = "Temperature [C°]")
    tool_salinity.plot_animation(      f"../images/salinity_animation_{date}.gif", ylabel = "Salinity [ppt]")
    tool_oxygen.plot_animation(          f"../images/oxygen_animation_{date}.gif", ylabel = "Oxygen [mmol/m3]")
    tool_chlorophyll.plot_animation(f"../images/chlorophyll_animation_{date}.gif", ylabel = "Chlorophyll [mmol/m3]")
    tool_klong.plot_animation(            f"../images/klong_animation_{date}.gif", ylabel = "K-Long [-]")
    tool_kshort.plot_animation(          f"../images/kshort_animation_{date}.gif", ylabel = "K-Short [-]")
    


