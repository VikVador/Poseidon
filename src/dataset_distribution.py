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
# A tool to check datasets distributions (joint, marginal)
#
import os
import sys
import xarray
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plots animation
from matplotlib.animation import FuncAnimation

# Custom library
from dataset import BlackSea_Dataset


class BlackSea_Dataset_Distribution():
    r"""A simple distribution analyzer for Black Sea dataset"""

    def __init__(self, dataloader: BlackSea_Dataset, datasets:list, year_start: int, year_end: int, month_start: int, month_end: int, subpopulation_percentage = 10):
        super().__init__()

        # Names of all the datasets
        self.datasets_name = ['Oxygen [mmol/m3]', 'Temperature [C°]', 'Salinity [ppt]', 'Chlorophyll [mmol/m3]', 'K-long [-]', 'K-short [-]']

        # Security
        self._validate_datasets_length(datasets)
        print("--- WARNING (Distribution) - Be sure to have the 'oxygen' dataset at index 0 in your list")

        # Storing the datasets and time range
        self.datasets    = datasets
        self.year_start  = dataloader.year_start
        self.year_end    = dataloader.year_end
        self.month_start = dataloader.month_start
        self.month_end   = dataloader.month_end

        # Security checks for time range
        self._validate_time_range(year_start, year_end, month_start, month_end)

        # Calculate indices for data trimming
        index_start, index_end = self.calculate_indices(year_start, month_start, year_end, month_end)

        # Trim and flatten the datasets
        dataset_trimmed = [d[index_start:index_end].flatten() for d in self.datasets]

        # Remove invalid indices assuming oxygen is the first dataset
        invalid_indices = (dataset_trimmed[0] == 0.)
        dataset_trimmed = [d[~invalid_indices] for d in dataset_trimmed]

        # Randomly sample a subpopulation
        indices = random.sample(range(0, dataset_trimmed[0].shape[0]), int(dataset_trimmed[0].shape[0] * subpopulation_percentage / 100))

        # Create a dictionary of the shape - "name of the variable": trimmed data
        data_dc = {a: b[indices] for a, b in zip(self.datasets_name, dataset_trimmed)}

        # Create a Pandas DataFrame
        self.df = pd.DataFrame(data_dc)

    def _validate_datasets_length(self, datasets):
        r"""Security (1) - Makes sure all the datasets were given as input, none are missing"""
        assert len(self.datasets_name) == len(datasets), f"ERROR (Distribution, _validate_datasets_length) - The datasets list is missing a dataset ({len(datasets)} < {len(self.datasets_name)})"

    def _validate_time_range(self, start, end, month_start, month_end):
        r"""Security (2) - Makes sure all that the given period time is available, i.e. looks if from wha was loaded the subdataset is available"""
        assert 0 <= start <= 9 and 0 <= end <= 9,                                 f"ERROR (Distribution, _validate_time_range) - Incorrect starting or ending year ({start}, {end})"
        assert 0 <= month_start <= 12 and 0 <= month_end <= 12,                   f"ERROR (Distribution, _validate_time_range) - Incorrect starting or ending month ({month_start}, {month_end})"
        assert start <= end and self.year_start <= start <= self.year_end <= end, f"ERROR (Distribution, _validate_time_range) - Incorrect years ({start} <= {end})"

    def calculate_indices(self, start_year, start_month, end_year, end_month):
        r"""Used to compute the indices for slicing the original dataset given the time period as input"""

        # Number of days in each month
        days_of_the_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Computing starting and ending indices
        index_s_month = sum(days_of_the_month[i] for i in range(self.month_start, start_month))
        index_e_month = sum(days_of_the_month[i] for i in range(1, end_month)) if start_year != end_year else sum(days_of_the_month[i] for i in range(start_month, end_month))
        index_s_year = (start_year - self.year_start) * 365
        index_e_year = (end_year   - start_year     ) * 365 + index_s_year

        # Computing final indices with error protection, i.e. if index too large we just take the lenght of dataset (could be improved)
        index_start = index_s_year + index_s_month
        index_end   = min(index_e_year + index_e_month, self.datasets[0].shape[0] - 1)

        return index_start, index_end

    def plot_marginal(self, save: bool = False, file_name: str = "marginal.png"):
        r"""Used to plot the marginal distribution of all physical variables, i.e. average over a given period"""

        # Set up the seaborn theme
        sns.set_theme()

        # Define a color palette with a different color for each variable
        colors = sns.color_palette("deep", n_colors = len(self.df.columns))

        # Create a 2 by 3 grid of subplots
        fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 10))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Plot histograms for each variable in the grid
        for i, (col, ax, color) in enumerate(zip(self.df.columns, axes, colors)):
            sns.histplot(self.df[col], kde = True, ax = ax, color = color, bins = 100, element = "bars", linewidth = 2)
            ax.set_ylabel('')
            ax.set_yticks([])

        # Adjust layout
        plt.tight_layout()

        # Saving the plot (if asked)
        if save:
            plt.savefig(f"{file_name}")
        else:
            plt.show()

    def plot_joint(self, save: bool = False, file_name: str = "joint.png"):
        r"""Used to plot the marginal distribution of all physical variables, i.e. average over a given period"""

        # Set up the seaborn theme
        sns.set_style("darkgrid", {'axes.grid': False})

        # Creation of the grid to plot joints and marginals
        g = sns.PairGrid(self.df, diag_sharey = False, corner=True)

        # Map lower plots with adjusted y-axis limits
        g.map_lower(sns.kdeplot, cmap = 'viridis')

        # Map diagonal plots with adjusted y-axis limits
        for i, col in enumerate(self.df.columns):
            g.map_diag(sns.kdeplot, color = "black", fill = True)

            # Set y-axis limit for diagonal plots
            y_axis_limit = g.diag_axes[i].get_ylim()
            g.diag_axes[i].set_ylim(y_axis_limit)

        # Save or show the plot
        if save:
            g.savefig(file_name)
        else:
            plt.show()