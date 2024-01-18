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
from metrics              import BlackSea_Metrics
from neural_networks      import FCNN

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
    splitting       = kwargs['Splitting']
    resolution      = kwargs['Resolution']
    windows_inputs  = kwargs['Window (Inputs)']
    windows_outputs = kwargs['Window (Output)']
    depth           = kwargs['Depth']
    architecture    = kwargs['Architecture']
    learning_rate   = kwargs['Learning Rate']
    kernel_size     = kwargs['Kernel Size']
    batch_size      = kwargs['Batch Size']
    nb_epochs       = kwargs['Epochs']

    # Security
    assert 0 < len(inputs), f"ERROR (main) - At least one input must be given"
    for i in inputs:
        assert i in ["temperature", "salinity", "chlorophyll", "kshort", "klong"], f"ERROR (main) - Unknown input {i}"

    # ------- Loading the data -------
    Dataset_physical = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "grid_T")
    Dataset_bio      = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "ptrc_T")

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

    # Stores the output, i.e. oxyen bottom values (here its everywhere, we are not limited to values on the continental shelf of ~120m)
    data_oxygen = Dataset_bio.get_oxygen_bottom(depth = depth)

    # Loading the black sea mask
    BS_mask = Dataset_physical.get_blacksea_mask(depth = depth)

    # ------- Preparing the data -------
    BSD_loader = BlackSea_Dataloader(x = input_datasets,
                                     y = data_oxygen,
                                  mask = BS_mask,
                                  mode = splitting,
                            resolution = resolution,
                                window = windows_inputs,
                            window_oxy = windows_outputs,
                        deoxy_treshold = 63,
                         datasets_size = [0.6, 0.3],
                                  seed = 2701)

    # Retreiving the individual dataloader
    dataset_train      = BSD_loader.get_dataloader("train",      batch_size = batch_size)
    dataset_validation = BSD_loader.get_dataloader("validation", batch_size = batch_size)
    dataset_test       = BSD_loader.get_dataloader("test",       batch_size = batch_size)

    # ------------------------------------------
    #                   Training
    # ------------------------------------------
    #
    # ------- WandB -------
    wandb.init(project = "esa-blacksea-deoxygenation-emulator-V1", config = kwargs)

    # ------ Environment ------
    def to_device(data, device):
        r"""Function to move tensors to GPU if available"""
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking = True)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting up training environment
    neural_net = FCNN(inputs = len(input_datasets), outputs =  windows_outputs, kernel_size = kernel_size)
    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(neural_net.parameters(), lr=learning_rate)

    # Pushing the model to the correct device
    neural_net.to(device)

    # Normalized oxygen treshold
    norm_oxy = BSD_loader.get_normalized_deoxygenation_treshold()

    for epoch in range(nb_epochs):

        # Information over terminal (1)
        print("-- Epoch: ", epoch, " --")

        # Used to compute the average training loss
        training_loss = 0.0

        # ----- TRAINING -----
        for x, y in dataset_train:

            # Moving data to the correct device
            x, y = to_device(x, device), to_device(y, device)

            # Forward pass, i.e. prediction of the neural network
            pred = neural_net.forward(x)

            # Computing the loss, i.e. the value -1 in the ground truth corresponds to the land !
            loss = criterion(pred[y != -1], y[y != -1])

            # Sending the loss to wandDB
            wandb.log({"Loss (T)": loss.detach().item()})

            # Accumulating the loss
            training_loss += loss.detach().item()

            # Reseting the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

        # Information over terminal (2)
        print("Loss (Training, Averaged over batch): ", training_loss / len(dataset_train))

        # Sending the loss to wandDB
        wandb.log({"Loss (T, AOB): ": training_loss / len(dataset_train)})

        # ----- VALIDATION -----
        with torch.no_grad():

            # Used to compute the average validation loss
            validation_loss = 0.0

            # Stores the number of valid samples
            valid_samples = 0

            # Stores the metrics (regression and classification)
            metrics_results = list()

            for x, y in dataset_validation:

                # Moving data to the correct device
                x, y = to_device(x, device), to_device(y, device)

                # Forward pass, i.e. prediction of the neural network
                pred = neural_net.forward(x)

                # Computing the loss, i.e. the value -1 in the ground truth corresponds to the land !
                loss = criterion(pred[y != -1], y[y != -1])

                # Sending the loss to wandDB the loss
                wandb.log({"Loss (V)": loss.detach().item()})

                # Accumulating the loss
                validation_loss += loss.detach().item()

                # Used to compute the metrics
                metrics_tool = BlackSea_Metrics(y, pred, norm_oxy)

                # Visual inspection (only on the first batch, i.e. to observe the same samples)
                if valid_samples == 0:
                    for i in range(5):
                        wandb.log({f"Sample {i}" : metrics_tool.plot_comparison(y, pred, norm_oxy, index_sample = i)})

                # Computing and storing results
                metrics_results.append(metrics_tool.compute_metrics())

                # Updating the number of valid samples
                valid_samples += metrics_tool.get_number_of_valid_samples()

            # Information over terminal (3)
            print("Loss (Validation, Averaged over batch): ", validation_loss / len(dataset_validation))

            # Sending the loss to wandDB
            wandb.log({"Loss (V, AOB): ": validation_loss / len(dataset_validation)})

            # Computing the average metrics, i.e. average per sample
            metrics_results = metrics_tool.compute_metrics_average(torch.tensor(metrics_results), valid_samples)

            # Names of the metrics
            metrics_results_names = metrics_tool.get_metrics_names()

            # Sending the metrics results to wandDB
            for i, result in enumerate(metrics_results):
                for j, r in enumerate(result):
                    wandb.log({f"{metrics_results_names[j]} ({i})": r})

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
# Creation of all the inputs combinations
input_list = ["temperature", "salinity", "chlorophyll", "kshort", "klong"]

# Generate all combinations
all_combinations = []
for r in range(1, len(input_list) + 1):
    all_combinations.extend(combinations(input_list, r))

# Convert combinations to lists
all_combinations = [list(combination) for combination in all_combinations]

# Storing all the information
arguments = {
    'month_start'     : [0],
    'month_end'       : [12],
    'year_start'      : [0],
    'year_end'        : [5],
    'Inputs'          : all_combinations,
    'Splitting'       : ["temporal", "spatial"],
    'Resolution'      : [64],
    'Window (Inputs)' : [1, 3, 7],
    'Window (Output)' : [7],
    'Architecture'    : ["FCNN"],
    'Learning Rate'   : [0.001],
    'Kernel Size'     : [3, 5, 7],
    'Batch Size'      : [64]
}

# Generate all combinations
param_combinations = list(product(*arguments.values()))

# Create a list of dictionaries
param_dicts = [dict(zip(arguments.keys(), combo)) for combo in param_combinations]

# ----
# Jobs
# ----
@job(array = len(param_dicts), cpus = 1, gpus = 1, ram = '64GB', time = '24:00:00', project = 'bsmfc', user = 'vmangeleer@uliege.be', type = 'FAIL')
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
        '--depth',
        help    = 'The maximum depth at which we will recover oxygen concentration, i.e. the depth of the continental shelf is at 120m',
        default = None)

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
        default = 5)

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
            "Splitting"       : args.splitting,
            "Resolution"      : args.resolution,
            "Window (Inputs)" : args.windows_inputs,
            "Window (Output)" : args.windows_outputs,
            "Depth"           : args.depth,

            # Training
            "Architecture"    : args.architecture,
            "Learning Rate"   : args.learning_rate,
            "Kernel Size"     : args.kernel_size,
            "Batch Size"      : args.batch_size,
            "Epochs"          : args.epochs
        }

        # Launching the main
        main(**arguments)

        # Information over terminal
        print("Done")