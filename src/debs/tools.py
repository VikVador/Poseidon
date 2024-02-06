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
import os
import cv2
import numpy as np

def to_device(data, device):
    r"""Moves tensors to a chosen device (CPU or GPU)"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

def get_data_path(folder:str):
    r"""Checks which path to use to get the data, i.e. if the folder is in the local version or the scratch version"""

    # Cluster OR local
    if os.path.exists("../../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS_BIO/"):
        return f"../../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS_BIO/{folder}/"
    if os.path.exists("../../data/"):
        return f"../../data/{folder}/"

    print("ERROR (get_data_path) - No path found")
    exit()

def get_mesh_path():
    r"""Checks which path to use to get the mesh, i.e. if the folder is in the local version or the scratch version"""

    # Cluster OR local
    if os.path.exists("../../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS/"):
        return f"../../../../../../../scratch/acad/bsmfc/nemo4.2.0/BSFS/mesh_mask.nc_new59_CMCC_noAzov"
    if os.path.exists("../../data/"):
        return f"../../data/mesh_mask.nc_new59_CMCC_noAzov"

    print("ERROR (get_mesh_path) - No path found")
    exit()

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

# Used to display a simple progress bar while training for 1 epoch
def progressBar(loss_training, loss_validation, learning_rate, estimated_time_epoch, nb_epoch_left, percent, width = 15):

    # Setting up the useful information
    left            = width * percent // 100
    right           = width - left
    tags            = "-" * int(left)
    spaces          = " " * int(right)
    percents        = f"{percent:.2f} %"
    loss_training   = f"{loss_training * 1:.3f}"
    loss_validation = f"{loss_validation * 1:.3f}"
    learning_rate   = f"{learning_rate[-1] * 1:.6f}"

    # Total timing left in seconds
    total_time_left = nb_epoch_left * estimated_time_epoch

    # Contains the unit of the timer and message to be printed
    timer_unit, estimated_time_total = str(), str()

    # Conversion to hours
    if total_time_left > 3600:
        total_time_left      = total_time_left/3600
        timer_unit           = "h"
        estimated_time_total = f"{total_time_left:.0f} {timer_unit}"

    # Conversion to minutes
    elif total_time_left > 60:
        total_time_left      = total_time_left/60
        timer_unit           = "min"
        estimated_time_total = f"{total_time_left:.0f} {timer_unit}"

    # A few seconds left
    else:
        timer_unit           = "s"
        estimated_time_total = f"{total_time_left:.0f} {timer_unit}"

    # Nothing to print for now
    if total_time_left == 0:
        estimated_time_total = " - "

    # Displaying a really cool progress bar !
    print("\r[", tags, spaces, "] - ", percents, " | Loss (Training) = ", loss_training, " | Loss (Validation) = ", loss_validation,
          " | LR : ", learning_rate, " | Time : ", estimated_time_total, " | ", sep = "", end = "", flush = True)