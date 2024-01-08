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
# A script to train a neural network
#
#   Dawgz = False : compute the distributions over a given time period given by the user as arguments
#
#   Dawgz = True  : compute the distributions over all the possible time periods
#
import wandb
import argparse

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Custom libraries
from dataset              import BlackSea_Dataset
from dataloader           import BlackSea_Dataloader
from neural_networks      import FCNN

# Dawgz library (used to parallelized the jobs)
from dawgz import job, schedule

# ---------------------------------------------------------------------
#
#                              MAIN FUNCTION
#
# ---------------------------------------------------------------------
def main(**kwargs):

    # ------- Arguments -------
    start_month     = kwargs['month_start']
    end_month       = kwargs['month_end']
    start_year      = kwargs['year_start']
    end_year        = kwargs['year_end']
    inputs          = kwargs['Inputs']
    splitting       = kwargs['Splitting']
    resolution      = kwargs['Resolution']
    windows_inputs  = kwargs['Window (Inputs)']
    windows_outputs = kwargs['Window (Output)']
    architecture    = kwargs['Architecture']
    learning_rate   = kwargs['Learning Rate']
    kernel_size     = kwargs['Kernel Size']
    batch_size      = kwargs['Batch Size']

    # ------- Loading the data -------
    Dataset_physical = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "grid_T")
    Dataset_bio      = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "ptrc_T")

    # Retreives oxyen bottom values, i.e. here its everywhere (we are not limited to values on the continental shelf of ~120m)
    data_oxygen = Dataset_bio.get_oxygen_bottom()

    # Stores all the inputs
    input_datasets = list()

    # Retreives the different inputs
    if "temperature" in inputs:
        input_datasets.append(Dataset_physical.get_temperature())
    if "salinity" in inputs:
        input_datasets.append(Dataset_physical.get_salinity())
    if "chlorophyll" in inputs:
        input_datasets.append(Dataset_bio.get_chlorophyll())
    if "kshort" in inputs:
        input_datasets.append(Dataset_bio.get_light_attenuation_coefficient_short_waves())
    if "klong" in inputs:
        input_datasets.append(Dataset_bio.get_light_attenuation_coefficient_long_waves())

    # Loading the black sea mask
    BS_mask = Dataset_physical.get_blacksea_mask()

    # ------- Preparing the data -------
    BSD_loader = BlackSea_Dataloader(x = input_datasets,
                                     y = data_oxygen,
                                  mask = BS_mask,
                                  mode = splitting,
                            resolution = resolution,
                                window = windows_inputs,
                            window_oxy = windows_outputs)

    # Retreiving the individual dataloader
    dataset_train      = BSD_loader.get_dataloader("train",      batch_size = batch_size)
    dataset_validation = BSD_loader.get_dataloader("validation", batch_size = batch_size)
    dataset_test       = BSD_loader.get_dataloader("test",       batch_size = batch_size)

    # -------------------------------------------
    #                   TRAINING
    # -------------------------------------------
    #
    # Initialization of weights and biases
    wandb.init(project = "esa-blacksea-deoxygenation-emulator-V1", config = kwargs)

    # Setting up training environment
    neural_net = FCNN(inputs = len(input_datasets), kernel_size = kernel_size)
    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)

    # Going through epochs
    for epoch in range(25):

        # Going through the training set
        for x, y in dataset_train:

            # Forward pass, i.e. prediction of the neural network
            pred = neural_net.forward(x)

            # Computing the loss, i.e. the value -1 in the ground truth corresponds to the land !
            loss = criterion(pred[y != -1], y[y != -1])

            # Sending the loss to wandDB the loss
            wandb.log({"Loss (Training)": loss.detach().item()})

            # Reseting the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

    # Finishing the run
    wandb.finish()


# ---------------------------------------------------------------------
#
#                                  DAWGZ
#
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
#
#                                  MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # ------- Parsing the command-line arguments -------
    #
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE:      python script_training.py --start_year    <X>
                                          --end_year      <X>
                                          --start_month   <X>
                                          --end_month     <X>
                                          --dawgz         <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--start_year',
        help    = 'Starting year to collect data',
        type    = int,
        default = 0)

    parser.add_argument(
        '--end_year',
        help    = 'Ending year to collect data',
        type    = int,
        default = 0)

    parser.add_argument(
        '--start_month',
        help    = 'Starting month to collect data',
        type    = int,
        default = 1)

    parser.add_argument(
        '--end_month',
        help    = 'Ending month to collect data',
        type    = int,
        default = 2)

    parser.add_argument(
        '--inputs',
        help    = 'Inputs to be used for the training, e.g. temperature, salinity, ...',
        nargs   = '+',
        type    = str)

    parser.add_argument(
        '--splitting',
        help    = 'Defines how to split the data into training, validation and testing sets.',
        type    = str,
        default = 'spatial',
        choices = ['spatial', 'temporal'])

    parser.add_argument(
        '--resolution',
        help    = 'The resolution of the input data, i.e. the size of the patches',
        type    = int,
        default = 32)

    parser.add_argument(
        '--windows_inputs',
        help    = 'The number of days of data to be used as input',
        type    = int,
        default = 1)

    parser.add_argument(
        '--windows_outputs',
        help    = 'The number of days to predict, i.e. the oxygen forecast for the next days',
        type    = int,
        default = 1)

    parser.add_argument(
        '--architecture',
        help    = 'The neural network architecture to be used',
        type    = str,
        default = 'FCNN',
        choices = ["FCNN"])

    parser.add_argument(
        '--learning_rate',
        help    = 'The learning rate used for the training',
        type    = float,
        default = 0.01)

    parser.add_argument(
        '--batch_size',
        help    = 'The batch size used for the training',
        type    = int,
        default = 64)

    parser.add_argument(
        '--dawgz',
        help    = 'Determine if the script is run with dawgz or not',
        type    = str,
        default = "False",
        choices = ['True', 'False'])

    # ----- FCNN ------
    parser.add_argument(
        '--kernel_size',
        help    = 'The size of the kernel used for the convolutional layers',
        type    = int,
        default = 3)

    # Retrieving the values given by the user
    args = parser.parse_args()

    # ------- Running with dawgz -------
    if args.dawgz == "True":

        # Information over terminal
        print("Running with dawgz")

        # Running the jobs
        # schedule(compute_distribution, name = 'nn_training', backend = 'slurm', export = 'ALL')

    # ------- Running without dawgz -------
    else:

        # Information over terminal
        print("Running without dawgz")

        # Storing all the information
        arguments = {

            # Temporal Information
            'month_start'     : args.start_month,
            'month_end'       : args.end_month,
            'year_start'      : args.start_year,
            'year_end'        : args.end_year,

            # Datasets
            "Inputs"          : args.inputs,
            "Splitting"       : args.splitting,
            "Resolution"      : args.resolution,
            "Window (Inputs)" : args.windows_inputs,
            "Window (Output)" : args.windows_outputs,

            # Training
            "Architecture"    : args.architecture,
            "Learning Rate"   : args.learning_rate,
            "Kernel Size"     : args.kernel_size,
            "Batch Size"      : args.batch_size
        }

        # Launching the main
        main(**arguments)

        # Information over terminal
        print("Done")