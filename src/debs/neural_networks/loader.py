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
# A simple tool to load neural networks
#
import numpy as np

# Custom libraries
from .fcnn     import FCNN
from .unet     import UNET
from .average  import AVERAGE


def load_neural_network(architecture : str, data_output : np.array, device : str, kwargs : dict):
    r"""Loads a neural network of a given architecture"""

    # Security
    assert architecture in ["FCNN", "UNET", "AVERAGE"], f"Architecture {architecture} not recognized"

    # Extracting information
    inputs          = kwargs['Inputs']
    windows_inputs  = kwargs['Window (Inputs)']
    architecture    = kwargs['Architecture']
    scaling         = kwargs['Scaling']
    kernel_size     = kwargs['Kernel Size']

    # Determining the total number of inputs
    nb_inputs = 0
    for i in ["temperature", "salinity", "chlorophyll", "kshort", "klong"]:
        nb_inputs += 1 if i in inputs else 0

    # Final number of inputs
    nb_inputs = windows_inputs * nb_inputs + 4

    # Initialization of neural network and pushing it to device (GPU)
    if architecture == "FCNN":
        return FCNN(inputs = nb_inputs, scaling = scaling, kernel_size = kernel_size)

    elif architecture == "UNET":
        return UNET(inputs = nb_inputs, scaling = scaling)

    elif architecture == "AVERAGE":
        return AVERAGE(data_output = data_output, device = device, kwargs = kwargs)

    else:
        raise ValueError("Unknown architecture")