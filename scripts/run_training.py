r"""Scripts - Train a denoiser model."""

import argparse

from dawgz import job, schedule

# isort: split
from poseidon.parser import load_configuration
from poseidon.training import training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of a denoiser.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the .yml configuration file.",
    )
    parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging advancement of the training.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the configuration from the specified YAML file
    config_list = load_configuration(args.config)

    # Extract the cluster configuration from the loaded config
    config_cluster = config_list[0]["Cluster"]

    # Check if training should be done locally
    if not config_cluster["Dawgz"]:
        training(
            config_dataloader=config_list[0]["Dataloader"],
            config_backbone=config_list[0]["Backbone"],
            config_nn=config_list[0]["Neural Network"],
            config_training=config_list[0]["Training"],
            toy_problem=config_list[0]["Problem"]["Toy_problem"],
            wandb_mode="online" if args.use_wandb else "disabled",
        )

    # Proceed with cluster training if Dawgz is enabled
    else:

        @job(
            array=len(config_list),
            cpus=config_cluster["CPUS"],
            gpus=config_cluster["GPUS"],
            ram=config_cluster["RAM"],
            time=config_cluster["TIME"],
            partition=config_cluster["PARTITION"],
            account="bsmfc",
        )
        def training_neural_network(i: int):
            training(
                config_dataloader=config_list[i]["Dataloader"],
                config_backbone=config_list[i]["Backbone"],
                config_nn=config_list[i]["Neural Network"],
                config_training=config_list[i]["Training"],
                toy_problem=config_list[i]["Problem"]["Toy_problem"],
                wandb_mode="online" if args.use_wandb else "disabled",
            )

        # Schedule the defined jobs for execution on the cluster
        schedule(training_neural_network, name="POSEIDON-TRAINING", backend="slurm", export="ALL")
