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
from tools                    import *
from losses                   import compute_loss
from dataset                  import BlackSea_Dataset
from metrics                  import BlackSea_Metrics
from dataloader               import BlackSea_Dataloader
from neural_networks.loader   import load_neural_network
from neural_networks.FCNN     import FCNN
from neural_networks.UNET     import UNET
from neural_networks.AVERAGE  import AVERAGE

# Dawgz library (used to parallelized the jobs)
from dawgz import job, schedule

# Combinatorics
from itertools import product


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
    start_month      = kwargs['month_start']
    end_month        = kwargs['month_end']
    start_year       = kwargs['year_start']
    end_year         = kwargs['year_end']
    inputs           = kwargs['Inputs']
    problem          = kwargs['Problem']
    windows_inputs   = kwargs['Window (Inputs)']
    windows_outputs  = kwargs['Window (Output)']
    depth            = kwargs['Depth']
    hypoxia_treshold = kwargs['Hypoxia Treshold']
    architecture     = kwargs['Architecture']
    learning_rate    = kwargs['Learning Rate']
    batch_size       = kwargs['Batch Size']
    dataset_size     = kwargs['Dataset Size']
    nb_epochs        = kwargs['Epochs']

    # ------- Parameters -------
    #
    # Project name on Weights and Biases
    project_name = "esa-blacksea-deoxygenation-emulator-one-month"

    # Size of the different datasets
    size_training, size_validation = dataset_size[0], dataset_size[1]

    # Seed to fix randomness
    seed_of_chaos = 2701

    # ------- Data -------
    Dataset_phy = BlackSea_Dataset(start_year, end_year, start_month, end_month, variable = "grid_T")
    Dataset_bio = BlackSea_Dataset(start_year, end_year, start_month, end_month, variable = "ptrc_T")

    # Stores all the inputs
    input_datasets = list()

    # Loading the inputs
    for i in inputs:

        # Physical variables
        if i in ["temperature", "salinity"]:
            input_datasets.append(Dataset_phy.get_data(variable = i, type = "surface", depth = None))

        # Biogeochemical variables
        if i in ["chlorophyll", "kshort", "klong"]:
            input_datasets.append(Dataset_bio.get_data(variable = i, type = "surface", depth = None))

    # Loading the output
    data_oxygen = Dataset_bio.get_data(variable = "oxygen", type = "bottom", depth = depth)

    # Retrieving dimensions (Ease of comprehension)
    t, x_res, y_res = data_oxygen.shape

    # Loading Black Sea masks
    bs_mask             = Dataset_phy.get_mask(depth = None)                 # Only Land
    bs_mask_with_depth  = Dataset_phy.get_mask(depth = depth)                # Land and unobserved sea
    bs_mask_complete    = get_complete_mask(data_oxygen, bs_mask_with_depth) # Land, unobserved sea, oxygenated, switching and hypoxic regions

    # Loading bathymetry data (normalized between 0 and 1)
    bathy = Dataset_phy.get_depth() if "bathymetry" in inputs else None

    # Loading x mesh (x & y between 0/1 and -2 to resolution to be multiple of 2)
    mesh = Dataset_phy.get_mesh(x_res - 2, y_res - 2) if "mesh" in inputs else None

    # Retrieves the ratios of the different classes (Used to get insights about the data)
    ratio_oxygenated, ratio_switching, ratio_hypoxia = get_ratios(bs_mask_complete)

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
                                  seed = seed_of_chaos)

    # Retreiving the individual dataloader
    dataset_train      = BSD_loader.get_dataloader(type = "train",      bathy = bathy, mesh =  mesh, batch_size = batch_size)
    dataset_validation = BSD_loader.get_dataloader(type = "validation", bathy = bathy, mesh =  mesh, batch_size = batch_size)
    dataset_test       = BSD_loader.get_dataloader(type = "test",       bathy = bathy, mesh =  mesh, batch_size = batch_size)

    # Normalized oxygen treshold
    norm_oxy = BSD_loader.get_normalized_deoxygenation_treshold()

    # Number of batches in training set (used for averaging metrics over the batches)
    num_batches_train = BSD_loader.get_number_of_batches(type = "train", batch_size = batch_size)

    # ------------------------------------------
    #
    #                   TRAINING
    #
    # ------------------------------------------
    #
    # ------- WandB -------
    wandb.init(project = project_name, config = kwargs)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sending information about the dataset to WandB (1)
    wandb.log({"Dataset & Architecture/Dataset (Visualisation, Image, Regions)" : wandb.Image(get_complete_mask_plot(bs_mask_complete)),
               "Dataset & Architecture/Dataset (Visualisation, Image, Ratios)"  : wandb.Image(get_ratios_plot(data_oxygen, bs_mask_with_depth)),
               "Dataset & Architecture/Dataset (Visualisation, Video, Oxygen)"  : wandb.Video(get_video(data = data_oxygen), fps = 4),
               "Dataset & Architecture/Ratio Oxygenated"                        : ratio_oxygenated,
               "Dataset & Architecture/Ratio Switching"                         : ratio_switching,
               "Dataset & Architecture/Ratio Hypoxia"                           : ratio_hypoxia})

    # Initialization of neural network and pushing it to correct device
    neural_net = load_neural_network(architecture = architecture,
                                     data_output  = data_oxygen,
                                     device       = device,
                                     kwargs       = kwargs)

    # Sending information about the Neural Network
    wandb.log({"Dataset & Architecture/Trainable Parameters" : neural_net.count_parameters()})

    # Initialization of the optimizer and the loss function
    optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)

    # Information over terminal (1)
    project_title(kwargs)

    # Used to compute time left
    epoch_time = 0.0

    # Starting training !
    for epoch in range(nb_epochs):

        # Used to approximate time left for current epoch and in total
        start = time.time()

        # Used to store instantaneous loss and compute the average per batch (AOB) training loss
        training_loss        = 0.0
        training_batch_steps = 0

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

            # Computing the loss
            loss_t = compute_loss(y_pred = pred, y_true = y, problem = problem, kwargs = kwargs)

            # Information over terminal (2)
            progression(epoch = epoch,
                        number_epoch        = nb_epochs,
                        loss_training       = loss_t.detach().item(),
                        loss_training_aob   = 0,
                        loss_validation     = 0,
                        loss_validation_aob = 0)

            # Sending to wandDB
            wandb.log({"Training/Loss (T)": loss_t.detach().item()})

            # Accumulating the loss
            training_loss += loss_t.detach().item()

            # Updating epoch information
            training_batch_steps += 1

            # AverageNet - No backpropagation
            if architecture == "AVERAGE":
                continue

            # Reseting the gradients
            optimizer.zero_grad()

            # Backward pass
            loss_t.backward()

            # Optimizing the parameters
            optimizer.step()

            break

        # Information over terminal (3)
        progression(epoch = epoch,
                    number_epoch        = nb_epochs,
                    loss_training       = loss_t.detach().item(),
                    loss_training_aob   = training_loss / training_batch_steps,
                    loss_validation     = 0,
                    loss_validation_aob = 0)

        # Sending the loss to wandDB
        wandb.log({"Training/Loss (Training): ": training_loss / training_batch_steps})

        # ----- VALIDATION -----
        with torch.no_grad():

            # Used to store instantaneous loss and compute the average per batch (AOB) training loss
            validation_loss        = 0.0
            validation_batch_steps = 0

            for x, y in dataset_validation:

                # Moving data to the correct device
                x, y = to_device(x, device), to_device(y, device)

                # Forward pass, i.e. prediction of the neural network
                pred = neural_net.forward(x)

                # Computing the loss
                loss_v = compute_loss(y_pred = pred, y_true = y, problem = problem, kwargs = kwargs)

                # Information over terminal (4)
                progression(epoch = epoch,
                            number_epoch        = nb_epochs,
                            loss_training       = loss_t.detach().item(),
                            loss_training_aob   = training_loss / training_batch_steps,
                            loss_validation     = loss_v.detach().item(),
                            loss_validation_aob = 0)

                # Sending the loss to wandDB the loss
                wandb.log({"Training/Loss (V)": loss_v.detach().item()})

                # Accumulating the loss
                validation_loss += loss_v.detach().item()

                # Transforming to probabilities (not done in forward pass because BCEWithLogitsLoss does it for us)
                x = nn.Softmax(dim = 2)(pred) if problem == "classification" else pred

                # Moving to CPU
                pred, y = pred.cpu(), y.cpu()

                # Used to compute the metrics
                metrics_tool.compute_metrics(y_pred = pred, y_true = y)

                # Visual inspection (Only on the first batch)
                metrics_tool.compute_plots(y_pred = pred, y_true = y) if validation_batch_steps == 0 else None

                # Updating epoch information
                validation_batch_steps += 1

            # Information over terminal (5)
            progression(epoch = epoch,
                        number_epoch        = nb_epochs,
                        loss_training       = loss_t.detach().item(),
                        loss_training_aob   = training_loss / training_batch_steps,
                        loss_validation     = loss_v.detach().item(),
                        loss_validation_aob = validation_loss / validation_batch_steps)

            # Sending more information to wandDB
            wandb.log({"Training/Loss (Validation)": validation_loss / validation_batch_steps,
                       "Training/Epochs"           : nb_epochs - epoch})

            # ---------- WandB (Metrics & Plots) ----------
            #
            # Getting results of each metric (averaged over each batch)
            results, results_name = metrics_tool.get_results()

            # Sending these results to wandDB
            for d, day_results in enumerate(results):
                for i, result in enumerate(day_results):

                    # Metric with corresponding forecasted day (Only if more than 1 day is forecasted)
                    m_name = results_name[i] + " D(" + str(d) + ")" if windows_outputs > 1 else results_name[i]

                    # Logging
                    wandb.log({f"Metrics/{m_name}" : result})

            # Getting the plots
            plots, plots_name = metrics_tool.get_plots()

            # Sending the plots to wandDB
            for plot, name in zip(plots, plots_name):

                # Logging
                wandb.log({f"Visualization/{name}" : wandb.Image(plot)})
                pass

        # Updating timing
        epoch_time = time.time() - start

        # Sending time left to WandB
        wandb.log({"Training/Time Left": (nb_epochs - epoch) * epoch_time})

    # Finishing the Weight and Biases run
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
# Creation of all the inputs combinations (Example : ["temperature"], ["salinity"], ["chlorophyll"], ["kshort"], ["klong"])
input_list = [["temperature"]]

# Storing all the information
arguments = {
    'month_start'     : [0],
    'month_end'       : [12],
    'year_start'      : [0],
    'year_end'        : [0],
    'Inputs'          : input_list,
    'Problem'         : ["regression", "classification"],
    'Window (Inputs)' : [1],
    'Window (Output)' : [1],
    'Depth'           : [200],
    'Architecture'    : ["FCNN", "UNET", "AVERAGE"],
    'Scaling'         : [1],
    'Kernel Size'     : [3],
    'Loss Weights'    : [[1, 1], [1, 2], [1, 5], [1, 10]],
    'Learning Rate'   : [0.001],
    'Batch Size'      : [64],
    'Epochs'          : [20]
}

# Generate all combinations
param_combinations = list(product(*arguments.values()))

# Create a list of dictionaries
param_dicts = [dict(zip(arguments.keys(), combo)) for combo in param_combinations]

# ----
# Jobs
# ----
@job(array = len(param_dicts), cpus = 1, gpus = 1, ram = '128GB', time = '1:00:00', project = 'bsmfc', partition = "debug-gpu", user = 'vmangeleer@uliege.be', type = 'FAIL')
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
        '--hypoxia_treshold',
        help    = 'The concentration treshold used to detect hypoxia, i.e. here it is set to 63 mmol/m^3 (see M. Grégoire & al. 2017, Biogeosciences, 14, 1733-1752)',
        type    = int,
        default = 63)

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
        '--kernel_size',
        help    = 'The size of the kernel used for the convolutional layers',
        type    = int,
        default = 3)

    parser.add_argument(
        '--loss_weights',
        help    = "The size of the weights for classification loss, i.e. if [1, 10] (= [C0, C1]) the error made on hypoxia (= C1)" + \
                  "is 10 times more important than the error made on oxygenated regions",
        nargs   = '+',
        type    = int,
        default = [1, 1])

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
        '--dataset_size',
        help    = "The size of the dataset used for the training and validation datasets, i.e. for example [0.6, 0.3, inferred]",
        nargs   = '+',
        type    = float,
        default = [0.6, 0.3])

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

    # Retrieving the values given by the user
    args = parser.parse_args()

    # ------- Running with dawgz -------
    if args.dawgz == "True":

        # Information over terminal
        print("------------------------")
        print("----- Using Dawgz ------")
        print("------------------------\n\n")

        # Running the jobs
        schedule(train_model, name = 'neural_network_training', backend = 'slurm', export = 'ALL')

    # ------- Running without dawgz -------
    else:

        # Information over terminal
        print("----------------------------")
        print("----- Not Using Dawgz ------")
        print("----------------------------\n\n")


        # Storing all the information
        arguments = {

            # Temporal Information
            'month_start'      : args.start_month,
            'month_end'        : args.end_month,
            'year_start'       : args.start_year,
            'year_end'         : args.end_year,

            # Datasets
            "Inputs"           : args.inputs,
            "Problem"          : args.problem,
            "Window (Inputs)"  : args.windows_inputs,
            "Window (Output)"  : args.windows_outputs,
            "Depth"            : args.depth,
            "Hypoxia Treshold" : args.hypoxia_treshold,

            # Training
            "Architecture"     : args.architecture,
            "Scaling"          : args.scaling,
            "Kernel Size"      : args.kernel_size,
            "Loss Weights"     : args.loss_weights,
            "Learning Rate"    : args.learning_rate,
            "Batch Size"       : args.batch_size,
            "Dataset Size"     : args.dataset_size,
            "Epochs"           : args.epochs
        }

        # Adding boolean information about variable used (wandb cannot handle a list for parameters plotting)
        for v in ["temperature", "salinity", "chlorophyll", "kshort", "klong", "mesh", "bathymetry"]:
            arguments[v] = True if v in arguments['Inputs'] else False

        # Launching the main
        main(**arguments)

        # Information over terminal
        print("Done")