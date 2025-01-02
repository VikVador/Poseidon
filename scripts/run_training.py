r"""Script to launch a training pipeline."""

import argparse

from dawgz import job, schedule

# isort: split
from poseidon.training.parser import load_configuration
from poseidon.training.training import training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a training pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the training .yml configuration file.",
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

    # Loading every configurations
    list_of_configurations = load_configuration(args.config)

    # Extracting the cluster configuration
    config_cluster = list_of_configurations[0]["Cluster"]

    if args.backend == "async":
        training(
            **list_of_configurations[0].get("Training Pipeline"),
        )

    else:

        @job(array=len(list_of_configurations), **config_cluster)
        def launch_training_pipeline(i: int):
            training(
                **list_of_configurations[i].get("Training Pipeline"),
            )

        schedule(
            launch_training_pipeline,
            name="POSEIDON-TRAINING",
            backend="slurm",
            export="ALL",
        )
