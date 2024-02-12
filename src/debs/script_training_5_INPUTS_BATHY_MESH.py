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
# A script to train a neural network to become a oxygen concentration forecaster in the Black Sea.
#
import time
import wandb
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Custom libraries
from dataset                  import BlackSea_Dataset
from dataloader               import BlackSea_Dataloader
from metrics                  import BlackSea_Metrics
from neural_networks.FCNN     import FCNN
from neural_networks.UNET     import UNET
from neural_networks.AVERAGE  import AVERAGE
from tools                    import to_device, get_complete_mask, get_complete_mask_plot, get_ratios

# Dawgz library (used to parallelized the jobs)
from dawgz import job, schedule

# Combinatorics
from itertools import combinations, product


# ---------------------------------------------------------------------
#
#                              MAIN FUNCTION
#
# ---------------------------------------------------------------------
def main(**kwargs):

    # ------------------------------------------
    #               Initialization
    # ------------------------------------------
    #
    # ------- Arguments -------
    start_month     = kwargs['month_start']
    end_month       = kwargs['month_end']
    start_year      = kwargs['year_start']
    end_year        = kwargs['year_end']
    inputs          = kwargs['Inputs']
    problem         = kwargs['Problem']
    windows_inputs  = kwargs['Window (Inputs)']
    windows_outputs = kwargs['Window (Output)']
    depth           = kwargs['Depth']
    architecture    = kwargs['Architecture']
    scaling         = kwargs['Scaling']
    learning_rate   = kwargs['Learning Rate']
    kernel_size     = kwargs['Kernel Size']
    batch_size      = kwargs['Batch Size']
    nb_epochs       = kwargs['Epochs']

    # ------- Data -------
    Dataset_phy = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "grid_T")
    Dataset_bio = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "ptrc_T")

    # Loading the inputs
    input_datasets = list()
    for inp in inputs:
        if inp in ["temperature", "salinity"]:
            input_datasets.append(Dataset_phy.get_data(variable = inp, type = "surface", depth = None))
        if inp in ["chlorophyll", "kshort", "klong"]:
            input_datasets.append(Dataset_bio.get_data(variable = inp, type = "surface", depth = None))

    # Loading the output
    data_oxygen = Dataset_bio.get_data(variable = "oxygen", type = "bottom", depth = depth)

    # Loading the black sea mask
    bs_mask             = Dataset_phy.get_mask(depth = None)
    bs_mask_with_depth  = Dataset_phy.get_mask(depth = depth)
    bs_mask_complete    = get_complete_mask(data_oxygen, bs_mask_with_depth)

    # Loading bathymetry data (normalized between 0 and 1)
    bathy = Dataset_phy.get_depth() if "bathymetry" in inputs else None

    # Loading the mesh (normalized x, y between 0 and 1)
    mesh = Dataset_phy.get_mesh(data_oxygen.shape[1] - 2, data_oxygen.shape[2] - 2) if "mesh" in inputs else None

    # Retrive the ratios of the different classes
    ratio_oxygenated, ratio_switching, ratio_hypoxia = get_ratios(bs_mask_complete)

    # Size of the training and validation sets (needed for AverageNet and test set size is inferred)
    size_training = 0.6
    size_validation = 0.3

    # Treshold for detecting hypoxia
    hypoxia_treshold = 63

    # ------- Preprocessing -------
    BSD_loader = BlackSea_Dataloader(x = input_datasets,
                                     y = data_oxygen,
                               bs_mask = bs_mask,
                    bs_mask_with_depth = bs_mask_with_depth,
                                  mode = problem,
                            window_inp = windows_inputs,
                            window_out = windows_outputs,
                      hypoxia_treshold = hypoxia_treshold,
                         datasets_size = [size_training, size_validation],
                                  seed = 2701)

    # Retreiving the individual dataloader
    dataset_train      = BSD_loader.get_dataloader("train",      bathy, mesh, batch_size = batch_size)
    dataset_validation = BSD_loader.get_dataloader("validation", bathy, mesh, batch_size = batch_size)
    dataset_test       = BSD_loader.get_dataloader("test",       bathy, mesh, batch_size = batch_size)

    # Normalized oxygen treshold
    norm_oxy = BSD_loader.get_normalized_deoxygenation_treshold()

    # Total number of batches in the training set (used for averaging metrics over the batches)
    num_batches_train = BSD_loader.get_number_of_batches(type = "train", batch_size = batch_size)

    # -------- AverageNet ----------
    average_output = None

    # If we use AverageNet, i.e. a NN that only predicts the average, we need this information
    if architecture == "AVERAGE":

        # Retrieving dimensions
        t = data_oxygen.shape[0]

        # Number of training samples
        train_samples = int(t * size_training)

        # ----- Regression ------
        if problem == "regression":

            # Determine the minimum and maximum values of the data
            min_value = np.nanmin(data_oxygen)
            max_value = np.nanmax(data_oxygen)

            # Rescale the data to ensure non-negative values
            average_output = data_oxygen + np.abs(min_value) if min_value < 0 else data_oxygen

            # Normalizing the data
            average_output = (average_output - min_value) / (max_value - min_value)

            # Average concentration
            average_output = torch.mean(torch.from_numpy(average_output[: train_samples, :-2, :-2]), dim = 0)

        # ----- Classification ------
        else:

            # Converting to classification
            average_output =  torch.from_numpy((data_oxygen[: train_samples, :-2, :-2] < hypoxia_treshold) * 1)

            # Summing over time, i.e. if total number of hypoxic days is greater than 50% of the time, then it is hypoxic
            average_output = (torch.sum(average_output, dim = 0) > train_samples // 2) * 1

            # Conversion to "probabilities", i.e. (t, x, y) to (t, c, x, y) with c = 0 no hypoxia, c = 1 hypoxia
            average_output = torch.stack([(average_output == 0) * 1, average_output]).float()

    # ------------------------------------------
    #                   Training
    # ------------------------------------------
    #
    # ------- WandB -------
    wandb.init(project = "esa-blacksea-deoxygenation-emulator-analysis", config = kwargs)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sending visual information about the dataset to WandB (1)
    # wandb.log({"Dataset" : wandb.Image(get_complete_mask_plot(bs_mask_complete))})

    # Sending information about the dataset to WandB (1)
    wandb.log({"Ratio Oxygenated" : ratio_oxygenated,
               "Ratio Switching"  : ratio_switching,
               "Ratio Hypoxia"    : ratio_hypoxia})

    # Number of inputs
    nb_inputs = len(input_datasets)
    nb_inputs = nb_inputs + 1 if "bathymetry" in inputs else nb_inputs
    nb_inputs = nb_inputs + 2 if "mesh"       in inputs else nb_inputs

    # Initialization of neural network and pushing it to device (GPU)
    neural_net = None

    if architecture == "FCNN":
        neural_net = FCNN(inputs      = nb_inputs,
                          outputs     = windows_outputs,
                          scaling     = scaling,
                          problem     = problem,
                          kernel_size = kernel_size)

    elif architecture == "UNET":
        neural_net = UNET(inputs      = nb_inputs,
                          outputs     = windows_outputs,
                          scaling     = scaling,
                          problem     = problem)

    elif architecture == "AVERAGE":
        neural_net = AVERAGE(average    = average_output,
                             outputs    = windows_outputs,
                             batch_size = batch_size,
                             device     = device)

    else:
        raise ValueError("Unknown architecture")

    # Sending information about the Neural Network
    wandb.log({"Trainable Parameters" : neural_net.count_parameters()})

    # Pushing to correct device
    neural_net.to(device)

    # Initialization of the optimizer and the loss function
    optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)
    criterion  = nn.MSELoss() if problem == "regression" else nn.BCELoss()

    # Used to compute time left
    epoch_time = 0.0

    # Starting training !
    for epoch in range(nb_epochs):

        # Information over terminal (1)
        print("\n") if epoch == 0 else print("")
        print("Epoch : ", epoch + 1, "/", nb_epochs, "\n")

        # Used to approximate time left for current epoch and in total
        start = time.time()

        # Used to store instantaneous loss and compute the average per batch (AOB) training loss
        training_loss = 0.0
        batch_steps   = 0

        # Used to compute our metrics
        metrics_tool = BlackSea_Metrics(mode = problem,
                                        mask = bs_mask_with_depth,
                                        mask_complete = bs_mask_complete,
                                        treshold = norm_oxy,
                                        number_of_batches = num_batches_train)

        # ----- TRAINING -----
        for x, y in dataset_train:

            # Moving data to the correct device
            x, y = to_device(x, device), to_device(y, device)

            # Forward pass, i.e. prediction of the neural network
            pred = neural_net.forward(x)

            # Determine the indices of the valid samples, i.e. inside the observed region (-1 is the masked region)
            indices = torch.where(y != -1)

            # Computing the loss
            loss = criterion(pred[indices], y[indices])

            # Information over terminal (2)
            print("Loss (T) = ", loss.detach().item())

            # Sending to wandDB
            wandb.log({"Loss (T)": loss.detach().item()})

            # Accumulating the loss
            training_loss += loss.detach().item()

            # No needs for AverageNet !
            if architecture != "AVERAGE":

                # Reseting the gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Optimizing the parameters
                optimizer.step()

            # Updating epoch information
            batch_steps += 1

        # Information over terminal (3)
        print("Loss (Training, Averaged over batch): ", training_loss / batch_steps)

        # Sending the loss to wandDB
        wandb.log({"Loss (T, AOB): ": training_loss / batch_steps})

        # ----- VALIDATION -----
        with torch.no_grad():

            # Used to store instantaneous loss and compute the average per batch (AOB) training loss
            validation_loss = 0.0
            batch_steps = 0

            for x, y in dataset_validation:

                # Moving data to the correct device
                x, y = to_device(x, device), to_device(y, device)

                # Forward pass, i.e. prediction of the neural network
                pred = neural_net.forward(x)

                # Determine the indices of the valid samples, i.e. inside the observed region (-1 is the masked region)
                indices = torch.where(y != -1)

                # Computing the loss
                loss = criterion(pred[indices], y[indices])

                # Information over terminal (4)
                print("Loss (V) = ", loss.detach().item())

                # Sending the loss to wandDB the loss
                wandb.log({"Loss (V)": loss.detach().item()})

                # Accumulating the loss
                validation_loss += loss.detach().item()

                # Used to compute the metrics
                metrics_tool.compute_metrics(y_pred = pred.cpu(), y_true = y.cpu())

                # Visual inspection (Only on the first batch)
                metrics_tool.compute_plots(y_pred = pred.cpu(), y_true = y.cpu()) if batch_steps == 0 else None

                # Updating epoch information
                batch_steps += 1

            # Information over terminal (5)
            print("Loss (Validation, Averaged over batch): ", validation_loss / batch_steps)

            # Sending more information to wandDB
            wandb.log({"Loss (V, AOB): ": validation_loss / batch_steps})
            wandb.log({"Epochs : ": nb_epochs - epoch})

            # ---------- WandB (Metrics & Plots) ----------
            #
            # Getting results of each metric (averaged over each batch)
            results, results_name = metrics_tool.get_results()

            # Sending these results to wandDB
            for d, day_results in enumerate(results):
                for i, result in enumerate(day_results):

                    # Current name of metric with corresponding day
                    m_name = results_name[i] + " D(" + str(d) + ")"

                    # Logging
                    wandb.log({m_name : result})

            # Getting the plots
            plots, plots_name = metrics_tool.get_plots()

            # Sending the plots to wandDB
            for plot, name in zip(plots, plots_name):

                # Logging
                wandb.log({name : wandb.Image(plot)})

        # Updating timing
        epoch_time = time.time() - start

    # Finishing the run
    wandb.finish()

# ---------------------------------------------------------------------
#
#                                  DAWGZ
#
# ---------------------------------------------------------------------
#
# -------------
# Possibilities
# -------------
# Creation of all the inputs combinations (Example : ["temperature"], ["salinity"], ["chlorophyll"], ["k_short"], ["k_long"])
input_list = [["temperature", "mesh", "bathymetry"], ["salinity", "mesh", "bathymetry"], ["chlorophyll", "mesh", "bathymetry"], ["k_short", "mesh", "bathymetry"], ["k_long", "mesh", "bathymetry"], ["temperature", "salinity", "chlorophyll", "k_short", "k_long", "mesh", "bathymetry"]]

# Storing all the information
arguments = {
    'month_start'     : [0],
    'month_end'       : [12],
    'year_start'      : [0],
    'year_end'        : [9],
    'Inputs'          : input_list,
    'Problem'         : ["regression", "classification"],
    'Window (Inputs)' : [1],
    'Window (Output)' : [1],
    'Depth'           : [150],
    'Architecture'    : ["FCNN", "UNET"],
    'Scaling'         : [4],
    'Learning Rate'   : [0.001],
    'Kernel Size'     : [3],
    'Batch Size'      : [16],
    'Epochs'          : [20]
}

# Generate all combinations
param_combinations = list(product(*arguments.values()))

# Create a list of dictionaries
param_dicts = [dict(zip(arguments.keys(), combo)) for combo in param_combinations]

# ----
# Jobs
# ----
@job(array = len(param_dicts), cpus = 1, gpus = 1, ram = '512GB', time = '6:00:00', project = 'bsmfc', partition = "ia", user = 'vmangeleer@uliege.be', type = 'FAIL')
def train_model(i: int):

    # Launching the main
    main(**param_dicts[i])

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
        default = 0)

    parser.add_argument(
        '--end_month',
        help    = 'Ending month to collect data',
        type    = int,
        default = 1)

    parser.add_argument(
        '--inputs',
        help    = 'Inputs to be used for the training, e.g. temperature, salinity, ...',
        nargs   = '+',
        type    = str,
        default = ["temperature"])

    parser.add_argument(
        '--problem',
        help    = 'Defines how to formulate the problem.',
        type    = str,
        default = 'regression',
        choices = ['regression', 'classification'])

    parser.add_argument(
        '--windows_inputs',
        help    = 'The number of days of data to be used as input',
        type    = int,
        default = 1)

    parser.add_argument(
        '--windows_outputs',
        help    = 'The number of days to predict, i.e. the oxygen forecast for the next days',
        type    = int,
        default = 4)

    parser.add_argument(
        '--depth',
        help    = 'The maximum depth at which we will recover oxygen concentration, i.e. the depth of the continental shelf is at ~ 150m',
        type    = int,
        default = 200)

    parser.add_argument(
        '--architecture',
        help    = 'The neural network architecture to be used',
        type    = str,
        default = 'FCNN',
        choices = ["FCNN", "UNET", "AVERAGE"])

    parser.add_argument(
        '--scaling',
        help    = 'A scaling factor to be used for the neural network architecture, i.e. it increases the capacity of the network',
        type    = int,
        default = 1)

    parser.add_argument(
        '--learning_rate',
        help    = 'The learning rate used for the training',
        type    = float,
        default = 0.001)

    parser.add_argument(
        '--batch_size',
        help    = 'The batch size used for the training',
        type    = int,
        default = 64)

    parser.add_argument(
        '--epochs',
        help    = 'The number of epochs used for the training',
        type    = int,
        default = 10)

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
        schedule(train_model, name = 'neural_network_training', backend = 'slurm', export = 'ALL')

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
            "Problem"         : args.problem,
            "Window (Inputs)" : args.windows_inputs,
            "Window (Output)" : args.windows_outputs,
            "Depth"           : args.depth,

            # Training
            "Architecture"    : args.architecture,
            "Scaling"         : args.scaling,
            "Learning Rate"   : args.learning_rate,
            "Kernel Size"     : args.kernel_size,
            "Batch Size"      : args.batch_size,
            "Epochs"          : args.epochs
        }

        # Adding boolean information about variable used (wandb cannot handle a list for parameters plotting)
        for v in ["temperature", "salinity", "chlorophyll", "kshort", "klong"]:
            arguments[v] = True if v in arguments['Inputs'] else False

        # Launching the main
        main(**arguments)

        # Information over terminal
        print("Done")