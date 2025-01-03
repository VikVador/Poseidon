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
    #
    # fmt: off
    # Initialization
    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_cluster = configs[0]["Cluster"]

    if args.backend == "async":
        training(
            **configs[0].get("Training Pipeline"),
        )

    else:
        @job(array=len(configs), **config_cluster)
        def launch_training_pipeline(i: int):
            training(
                **configs[i].get("Training Pipeline"),
            )

        schedule(
            launch_training_pipeline,
            name="POSEIDON-TRAINING",
            backend="slurm",
            export="ALL",
        )
