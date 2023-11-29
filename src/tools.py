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
import sys
import xarray
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

# Custom library
from dataset import BlackSea_Dataset


class BlackSea_Tools():
    r"""A simple toolbox for black sea dataset"""
    
    def __init__(self, dataloader: BlackSea_Dataset, dataset: np.array):
        super().__init__()

        # Number of timesteps in the simulation and dataloader
        self.time_steps = dataset.shape[0]
        self.dataloader = dataloader
            
        # Creating a 3D mask, i.e. time steps x mask matrix for ease of function use
        self.blacksea_mask_3D = np.repeat(dataloader.get_blacksea_mask()[None, ...], self.time_steps, axis = 0)

        # Creation of a masked version of the dataset, i.e. hiding all the land around the black sea
        self.dataset_masked = np.ma.array(dataset, mask = (self.blacksea_mask_3D == 0))
        
    def compute_mean(self):
        r"Computing mean value over each snapshot for all time steps"
        return self.dataset_masked.mean(axis = (1, 2))

    def compute_std(self):
        r"Computing standard deviation value over each snapshot for all time steps"
        return self.dataset_masked.std(axis = (1, 2))
        
    def compute_var(self):
        r"Computing variance value over each snapshot for all time steps"
        return self.dataset_masked.var(axis = (1, 2))

    def plot_line(self, variable: str = "Unknown [-]", save: bool = False, file_name: str = "result"):
        r"Used to make a line plot of the mean value of a given field"

        # Computing mean and standard deviation value of the field over the black sea
        mean = self.compute_mean()
        std  = self.compute_std()
        
        # Creation of days vector
        days = np.arange(1, self.time_steps + 1)
        months = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
        num_months = (self.dataloader.year_end - self.dataloader.year_start) * 12 + self.dataloader.month_end - self.dataloader.month_start + 1
        months_indices = [(self.dataloader.month_start - 1 + i) % 12 for i in range(num_months)]
        
        # Configuration of the plot
        sns.set_theme(style = "white")       
        plt.figure(figsize = (25, 6))
        plt.ylabel(variable, x = -20)
        plt.xticks([(i + i + 30) // 2 for i in range(0, self.time_steps, 31)], [months[i] for i in months_indices], fontsize = 12)

        # Adding vertical lines every 31 days
        for day in range(31, self.time_steps, 31):
            plt.axvline(x = day, color = 'gray', linestyle = '-', linewidth = 0.5)

        # Standard deviation vector
        std_bottom = mean - std
        std_top    = mean + std

        # Small correction for large spatial variations that should not be negative
        std_bottom[std_bottom < 0] = 0
        
        # Adding data
        plt.plot(days, mean)
        plt.fill_between(days, std_bottom, std_top, alpha=.1)

        # Saving the plot (if asked)
        if save:
            plt.savefig(f"../images/{file_name}.png")
        else:
            plt.show()
   
    def plot_treshold(self, treshold_above: str = "Oxygenated", treshold_under: str = "Deoxygenated", treshold: int = 63, hide: bool = False, save: bool = False, file_name: str = "result"):
        r"Used to make a ratio / binary plot of a given field"
        
        # Filtering the values, i.e., checking if a region is above the given treshold or not
        data_filtered = self.dataset_masked[:] > treshold

        # Contains the total number of pixels
        total_pixels = np.count_nonzero(self.blacksea_mask_3D[0] == 1, axis=(0, 1))

        # Computing ratios for each time step
        ratios_above_tresh = np.count_nonzero(data_filtered == 1, axis=(1, 2)) / total_pixels

        # Assuming you have an attribute 'days' that represents the time index
        days = np.arange(data_filtered.shape[0])

        # Creation of days vector
        days = np.arange(1, self.time_steps + 1)
        months = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
        num_months = (self.dataloader.year_end - self.dataloader.year_start) * 12 + self.dataloader.month_end - self.dataloader.month_start + 1
        months_indices = [(self.dataloader.month_start - 1 + i) % 12 for i in range(num_months)]

        # Creating a filled area plot for the ratio of oxygenated areas
        plt.figure(figsize=(25, 6))
        plt.fill_between(days, 0, ratios_above_tresh * 100, alpha=0.25, color="green")
        plt.fill_between(days, ratios_above_tresh * 100, 100, alpha=0.25, color="blue")
        plt.plot(days, ratios_above_tresh * 100, color="black")

        # Adding labels
        plt.ylabel('Ratio [%]')
        plt.xticks([(i + i + 30) // 2 for i in range(0, self.time_steps, 31)], [months[i] for i in months_indices], fontsize=12)

        # Adding vertical lines every 31 days and annotations
        if not hide:
            for day in range(31, self.time_steps, 31):
                plt.axvline(x = day, color = 'gray', linestyle = '-', linewidth = 0.5)
                plt.annotate(f'{ratios_above_tresh[day - 1] * 100:.2f}%', xy = (day, ratios_above_tresh[day - 1] * 100),
                             xytext = (5, -15), textcoords = 'offset points', ha = 'center', va = 'center', color = 'black')

        # Adding text annotations
        plt.text(0.02, 0.90, treshold_under, transform = plt.gca().transAxes, fontsize = 12, color = 'blue')
        plt.text(0.02, 0.10, treshold_above, transform = plt.gca().transAxes, fontsize = 12, color = 'green')

        # Setting y-axis limits
        plt.ylim(0, 100)

        # Setting x-axis limits
        plt.xlim(1, self.time_steps)

        # Saving the plot (if asked)
        if save:
            plt.savefig(f"../images/{file_name}.png")
        else:
            plt.show()

    def plot_animation(self, file_name:str = 'animation.gif', fps = 10, interval = 100, ylabel = None, cmap = 'viridis'):
        r"""Generate an GIF animation from a 3D NumPy matrix"""

        def update(frame):
            r"""Update function for the animation. Updates the plot for each frame"""
            img.set_array(self.dataset_masked[frame, :, :])
            return img,
            
        # Creation of the figure and axis
        fig, ax = plt.subplots()
        img = ax.imshow(self.dataset_masked[0, :, :], cmap = cmap)

        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove contour around the plot
        ax.set_frame_on(False)
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames = len(self.dataset_masked), interval = interval, blit = True)

        # Add colorbar to the right
        cbar = fig.colorbar(img, ax = ax, orientation = 'horizontal', fraction = 0.07, pad = 0.04,
                            norm = mcolors.Normalize(vmin = np.min(self.dataset_masked), vmax = np.max(self.dataset_masked)))

        # Add label to the colorbar
        if ylabel:
            cbar.set_label(ylabel)
            
        # Save the animation as an MP4 file
        ani.save(file_name, writer = 'pillow', fps = fps)
    
        # Display the plot
        plt.show()    