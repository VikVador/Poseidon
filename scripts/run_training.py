r"""Script to launch a training pipeline"""

import argparse

from dawgz import job, schedule

# isort: split
from poseidon.training.parser import load_configuration
from poseidon.training.training import training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch training pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the training .yml configuration file.",
    )
    parser.add_argument(
        "--use_wandb",
        "-w",
        action="store_true",
        help="Use Weights & Biases for logging advancement of the training.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args = parser.parse_args()
    wandb_mode = "online" if args.use_wandb else "disabled"

    # Loading all possible configurations
    list_of_configurations = load_configuration(args.config)
    config_cluster = list_of_configurations[0]["Cluster"]

    if args.backend == "async":
        training(
            **list_of_configurations[0]["Training Pipeline"],
            wandb_mode=wandb_mode,
        )

    else:

        @job(array=len(list_of_configurations), **config_cluster)
        def launch_training_pipeline(i: int):
            training(
                **list_of_configurations[i]["Training Pipeline"],
                wandb_mode=wandb_mode,
            )

        schedule(
            launch_training_pipeline,
            name="POSEIDON-TRAINING",
            backend="slurm",
            export="ALL",
        )
