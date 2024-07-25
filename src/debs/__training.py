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
# A script to launch a training in local or on a cluster based on a congifugration file (.yml)
#
import yaml
import argparse

# Custom libraries
from training import training

# Dawgz library (used to parallelized the jobs)
from dawgz import job, schedule

# Combinatorics
from itertools import product


if __name__ == "__main__":

    # Initialization of the parser
    parser = argparse.ArgumentParser()

    # Adding the arguments
    parser.add_argument(
        '--config',
        help    = 'Name of the configuration file',
        type    = str,
        default = "local")

    # Retrieving the values given by the user
    args = parser.parse_args()

    # Loading the configuration file
    with open(f"../../configs/{args.config}.yml", "r") as file:
        config_file =  yaml.safe_load(file)

    # Opening the configuration file
    configuration = config_file["config"]

    # Local
    if config_file["cluster"]["Dawgz"] == False:

        # Launching the main
        training(**configuration)

    # Cluster
    else:

        # Generating all possible combinations of parameters given in the configuration file
        configuration_combinations = list(product(*configuration.values()))

        # Create a list of dictionaries
        configurations_list = [dict(zip(configuration.keys(), combo)) for combo in configuration_combinations]

        # Creating the job for Dawgz
        @job(array     = len(configurations_list),
             cpus      = config_file["cluster"]["CPUS"],
             gpus      = config_file["cluster"]["GPUS"],
             ram       = config_file["cluster"]["RAM"],
             time      = config_file["cluster"]["TIME"],
             partition = config_file["cluster"]["PARTITION"],
             account   = 'bsmfc')
        def training_neural_network(i: int):

            # Launching the main
            training(**configurations_list[i])

        # Running the jobs
        schedule(training_neural_network, name = 'neural_network_training', backend = 'slurm', export = 'ALL')