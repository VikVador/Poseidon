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
# A script to compute the joint and marginal distributions of the Black Sea datasets each month.
#
import numpy             as np
import pandas            as pd
import seaborn           as sns
import matplotlib.pyplot as plt

from dataset             import BlackSea_Dataset
from matplotlib.patches  import Rectangle
from datetime            import datetime, timedelta

# ----------
# Parameters
# ----------
sampling = 150
years    = np.arange(1980, 2023, 1)
months   = np.arange(1, 13, 1)

# Used to keep track of the plot index
index = 1

for year in years:
    for month in months:

        # Information over terminal
        print(f"Processing year {year} and month {month}")

        # Loading the dataset
        BSD_dataset        = BlackSea_Dataset(year_start = year, year_end = year, month_start = month, month_end = month, data_type = "normalized")
        data_temperature   = BSD_dataset.get_data(variable = "temperature")
        data_salinity      = BSD_dataset.get_data(variable = "salinity")
        data_chlorophyll   = BSD_dataset.get_data(variable = "chlorophyll")
        data_kshort        = BSD_dataset.get_data(variable = "kshort")
        data_klong         = BSD_dataset.get_data(variable = "klong")
        data_oxygen        = BSD_dataset.get_data(variable = "oxygenall")
        mask               = BSD_dataset.get_mask()

        # Masking the land
        data_oxygen      = data_oxygen[:, mask == 1]
        data_temperature = data_temperature[:, mask == 1]
        data_salinity    = data_salinity[:, mask == 1]
        data_chlorophyll = data_chlorophyll[:, mask == 1]
        data_kshort      = data_kshort[:, mask == 1]
        data_klong       = data_klong[:, mask == 1]

        # Samplig random subsets of values for each day (faster for the plots)
        np.random.seed(42)
        x, y            = data_oxygen.shape
        sampled_indices = np.random.choice(y, size = sampling, replace = False)

        sampled_oxygen      = data_oxygen[:, sampled_indices]
        sampled_temperature = data_temperature[:, sampled_indices]
        sampled_salinity    = data_salinity[:, sampled_indices]
        sampled_chlorophyll = data_chlorophyll[:, sampled_indices]
        sampled_kshort      = data_kshort[:, sampled_indices]
        sampled_klong       = data_klong[:, sampled_indices]

        # Flattening everything
        sampled_oxygen = sampled_oxygen.flatten()
        sampled_temperature = sampled_temperature.flatten()
        sampled_salinity = sampled_salinity.flatten()
        sampled_chlorophyll = sampled_chlorophyll.flatten()
        sampled_kshort = sampled_kshort.flatten()
        sampled_klong = sampled_klong.flatten()

        # Removing values associated to 0 oxygen which is mostly found in the deep sea
        bad_values_indexes = sampled_oxygen == 0

        sampled_oxygen      = sampled_oxygen[~bad_values_indexes]
        sampled_temperature = sampled_temperature[~bad_values_indexes]
        sampled_salinity    = sampled_salinity[~bad_values_indexes]
        sampled_chlorophyll = sampled_chlorophyll[~bad_values_indexes]
        sampled_kshort      = sampled_kshort[~bad_values_indexes]
        sampled_klong       = sampled_klong[~bad_values_indexes]

        # Removing values associated to 0 oxygen which is mostly found in the deep sea
        bad_values_indexes = sampled_chlorophyll == 0

        sampled_oxygen      = sampled_oxygen[~bad_values_indexes]
        sampled_temperature = sampled_temperature[~bad_values_indexes]
        sampled_salinity    = sampled_salinity[~bad_values_indexes]
        sampled_chlorophyll = sampled_chlorophyll[~bad_values_indexes]
        sampled_kshort      = sampled_kshort[~bad_values_indexes]
        sampled_klong       = sampled_klong[~bad_values_indexes]

        # Creation of the panda dataframe for seaborn plot
        data_frame = pd.DataFrame({"Oxygen"     : sampled_oxygen,
                                   "Temperature": sampled_temperature,
                                   "Salinity"   : sampled_salinity,
                                   "Chlorophyll": sampled_chlorophyll,
                                   "Kshort"     : sampled_kshort,
                                   "Klong"      : sampled_klong})

        # Information over terminal
        print(f"Generating the plots ({index})")

        # Plotting
        g = sns.pairplot(data_frame, diag_kind = "kde", diag_kws= {'color': '#4B244A'}, plot_kws = {'color': '#DBB68F'}, corner = True, aspect = 1)
        g.map_lower(sns.kdeplot, levels = 5, color = ".5")

        # Fixing y axis
        g.axes[0,0].set_ylim((0,1))
        g.axes[1,0].set_ylim((0,1))
        g.axes[2,0].set_ylim((0,1))
        g.axes[3,0].set_ylim((0,0.25))
        g.axes[4,0].set_ylim((0,0.1))
        g.axes[5,0].set_ylim((0,0.1))
        g.axes[-1,0].set_xlim((0,1))
        g.axes[-1,1].set_xlim((0,1))
        g.axes[-1,2].set_xlim((0,1))
        g.axes[-1,3].set_xlim((0,0.25))
        g.axes[-1,4].set_xlim((0,0.1))
        g.axes[-1,5].set_xlim((0,0.1))

        # Reduce tick label size
        for ax in g.axes.flatten():
            if ax is not None:
                ax.xaxis.set_tick_params(labelsize=9)
                ax.yaxis.set_tick_params(labelsize=9)

        # Date
        date = f"{year}-{month}" if month >= 10 else f"{year}-0{month}"

        # Increase spacing between subplots
        plt.subplots_adjust(hspace=0.2, wspace=0.2)

        # Add title padded from g.axes[2, 2]
        g.axes[0,0].set_title(date, fontsize=20, pad=20)

        # Saving
        plot_filename = f"../../analysis/distributions/distribution_{index}.png"
        plt.savefig(plot_filename, bbox_inches = 'tight')

        # Incrementing the index
        index += 1

        # Clearing the plot
        plt.clf()
        plt.close()