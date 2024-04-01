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
# Contains different random tools used in the project (e.g. generating fake datasets for testing, loading local or cluster path, ...)
#
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


def get_mask_info():
    """Used to retrieve the mask information, i.e. possible paths (Cluster or local) and the mask name"""

    # Loading the mask information
    with open("../../information/mask.txt", "r") as f:
        mask_info = json.load(f)

    # Extracting the information
    return mask_info["cluster"], mask_info["local"], mask_info["mask"]

def get_data_info():
    """Used to retrieve the data information, i.e. possible paths (Cluster or local)"""

    # Loading the data information
    with open("../../information/data.txt", "r") as f:
        data_info = json.load(f)

    # Extracting the information
    return data_info["cluster"], data_info["local"]

def get_complete_mask(data: np.array, treshold: float, bs_mask_with_depth: np.array):
    r"""Used to retrieve a mask highliting the land, oxygenated, hypoxia and switching zones"""

    # Converting to classification
    oxygen = (data < treshold) * 1

    # Averaging, i.e. if = 1, always in hypoxia, if = 0, never in hypoxia and value in between means there is a switch
    oxygen = np.nanmean(oxygen, axis = 0)

    # Applying masks (1)
    oxygen[oxygen == 0]             = 0

    # Indices where hypoxia remains a constant
    indices_hypoxia = np.where(oxygen == 1)

    # Applying masks (2)
    oxygen[oxygen == 1]             = 0
    oxygen[0 < oxygen]              = 1
    oxygen[indices_hypoxia]         = 2
    oxygen[bs_mask_with_depth == 0] = np.nan

    return oxygen

def get_complete_mask_plot(mask: np.array):
    r"""Used to plot the complete mask of the black sea."""

    # Labels for each of the classes
    labels = ['Not Observed', 'Oxygenated', 'Switching', 'Hypoxia']
    colors = ['#d6d6d6', '#94bac8', '#dac87e', '#bd2714']
    values = [-1, 0, 1, 2]

    # Replace NaN values with a placeholder value
    mask_without_nans = np.nan_to_num(mask, nan = -1)
    unique_values = np.unique(mask_without_nans)

    # Create a custom colormap with discrete colors
    cmap   = mcolors.ListedColormap(colors)
    bounds = np.arange(-1.5, 2.6, 1)  # Define boundaries for each value
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    # Creation of the plot
    fig = plt.figure(figsize = (15, 10))

    # Plot the masked array using the custom colormap
    plt.imshow(mask_without_nans, cmap = cmap, norm = norm)
    plt.grid(alpha = 0.25)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # Create custom legend with colored patches using the custom colormap
    patches = [mpatches.Patch(color = cmap(norm(value)), label = label) for value, label in zip(values, labels)]
    plt.legend(handles = patches, loc = 'upper right')

    return fig

def get_ratios(mask: np.array):
    r"""Used to retrieve the ratios (in %) of the different classes in the mask."""

    # Total number of elements (only in the observed region)
    total = mask[0 <= mask].size

    # Number of regions that always remain oxygenated
    oxygenated = (mask[mask == 0].size / total) * 100

    # Number of regions that always remain hypoxia
    hypoxia = (mask[mask == 2].size / total) * 100

    # Number of regions that switch between hypoxia and oxygenated
    switching = (mask[mask == 1].size / total) * 100

    return oxygenated, switching, hypoxia

def get_ratios_plot(data: np.array, treshold: float, bs_mask_with_depth: np.array):
    r"""Used to plot a ratios, i.e. it shows the tendency of a given region to be more oxygenated or in hypoxia."""

    # Retrieving dimensions (Ease of comprehension)
    t, x, y = data.shape

    # Converting to classification
    oxygen = (data < treshold) * 1

    # Counting the number of ones, i.e. number of times Hypoxia has occured
    oxygen_hypoxia = np.sum(oxygen, axis=0)

    # Creating ratios, i.e. the closer to 1 the more oxygenated
    oxygen_hypoxia = (t - oxygen_hypoxia) / t

    # Hiding unobserved regions
    oxygen_hypoxia[bs_mask_with_depth == 0] = np.nan

    # Define a custom colormap where np.nan values are mapped to grey
    cmap = plt.cm.RdBu
    cmap.set_bad(color = '#d6d6d6')
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(oxygen_hypoxia, vmin=0, vmax=1, cmap=cmap)
    plt.grid(alpha=0.25)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # Customize colorbar ticks and labels
    cbar = plt.colorbar(fraction = 0.021, location = 'bottom', pad = 0.1)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Often in Hypoxia', 'Mixed', 'Often Oxygenated'])
    plt.tight_layout()

    return fig

# ------------- Terminal ----------------
def progression(epoch : int, number_epoch : int, loss_training : float, loss_training_aob : float, loss_validation : float, loss_validation_aob : float):
    r"""Used to display the progress of the training"""
    print(f"Epoch [{epoch + 1}/{number_epoch}] | Loss (T): {loss_training:.4f} | Loss (T, AOB): {loss_training_aob:.4f} | Loss (V) : {loss_validation:.4f}, | Loss (V, AOB) : {loss_validation_aob:.4f}")

def project_title(kwargs : dict):
    r"""Used to display information about the run"""
    print("-------------------------------------------------------")
    print("                                                       ")
    print("                    ESA - PROJECT                      ")
    print("                                                       ")
    print("          BLACK SEA DEOXYGENATION EMULATOR             ")
    print("                                                       ")
    print("-------------------------------------------------------")
    print("                                                       ")
    for k, v in kwargs.items():
        print(f"- {k} : {v}")
    print("\n-----------------")
    print("Emulator Training")
    print("-----------------")

# ------------- Others ----------------
def to_device(data, device):
    r"""Moves tensors to a chosen device (CPU or GPU)"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

def get_video(data: np.array):
    r"""Used to transform array into a video"""

    # Assuming 'tensor' is your tensor of shape (t, x, y)
    tensor = torch.from_numpy(data)

    # Reshape the tensor to add a channel dimension
    tensor_reshaped = torch.unsqueeze(tensor, dim=1)

    # Expand the tensor along the new channel dimension
    tensor_expanded = tensor_reshaped.expand(-1, 3, -1, -1)

    # Returning tensor to numpy (Needed by WandB)
    return tensor_expanded.numpy()

def generateFakeDataset(number_of_variables: int = 5, number_of_samples: int = 14, oxygen : bool = False, resolution : int = 64, resolution_snapshot : tuple = (258, 258)):
    r"""Used to generate a list of fake datasets (numpy arrays) for testing purposes, i.e. each zone will be named to become easily recognizable"""

    # Stores all the fake samples
    list_fake_samples = [[] for i in range(number_of_variables)]

    # A simple list of letters to define the variables
    list_variables = ["Temperature", "Salinity", "Chlorophyll", "K-short", "K-long"] if oxygen == False else ["Oxygen"]

    def add_text_to_image(image: np.array, text: str, position: tuple, font_size: int = 0.45, thickness: int = 1):
        r"""Used to add some text on image"""
        return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,0), thickness, cv2.LINE_AA)

    # Creation of the fake samples
    for v in range(number_of_variables):
        for s in range(number_of_samples):

            # Creation of a snapshot
            snapshot = np.ones(shape = resolution_snapshot)

            # Indexes used for the grid
            index_i = int((snapshot.shape[0])  / resolution)
            index_j = int((snapshot.shape[1])  / resolution)

            # Adding horizontal lines
            for i in range(0, index_i):
                snapshot[i * resolution, :] = 0

            # Adding vertical lines
            for j in range(0, index_j):
                snapshot[:, j * resolution] = 0

            # Index of the space
            space_index = 1

            # Adding dummy zone notations
            for i in range(1, index_i + 1):
                for j in range(1, index_j + 1):

                    # Computing the middle of the zone
                    x      = int(j * resolution - resolution * (4/5))
                    y_step = int(i * resolution - resolution * (0.5/3))
                    y_name = int(i * resolution - resolution * (1/2))
                    y_var  = int(i * resolution - resolution * (2.5/3))

                    # Adding the name of the variable
                    snapshot = add_text_to_image(snapshot, f"{list_variables[v]}", (x, y_name))

                    # Adding the time step
                    snapshot = add_text_to_image(snapshot, f"t = {s}", (x, y_step))

                    # Adding the zone notation
                    snapshot = add_text_to_image(snapshot, f"Region -{space_index}-", (x, y_var))

                    # Updating the space index
                    space_index = space_index + 1

            # Adding the snapshot to the list of fake samples
            list_fake_samples[v].append(snapshot)

    # Creates a list of numpy arrays
    return [np.array(s) for s in list_fake_samples]

def crop(data: np.array, factor: int = 2):
    """Used to remove the borders of the data"""

    # Retrieving dimensions
    data_shape = data.shape

    # Croping the data
    return data[:-factor, :-factor] if len(data_shape) == 2 else data[:, :-factor, :-factor]

def crop_debug(data: np.array, factor: int = 2):
    """Used to remove the borders of the data (debugging purposes)"""

    # Retrieving dimensions
    data_shape = data.shape

    # Cropping indices
    minx = 128
    maxx = 156
    miny = 128
    maxy = 156

    # Croping the data
    return data[minx : maxx, miny : maxy] if len(data_shape) == 2 else data[:, minx : maxx, miny : maxy]