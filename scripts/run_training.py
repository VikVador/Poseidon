r"""Scripts - Train a denoiser model."""

import argparse
import yaml

from dawgz import Job, schedule

# isort: split
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
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.Loader)

    def dawgz_training():
        training(
            config_dataloader=config["Dataloader"],
            config_backbone=config["Backbone"],
            config_nn=config["Neural Network"],
            config_training=config["Training"],
            toy_problem=config["Problem"]["Toy_problem"],
        )

    # TO BE CHANGED

    schedule(
        Job(
            dawgz_training,
            cpus=8,
            gpus=1,
            mem="128GB",
            name="POSEIDON-TRAINING",
            time="00-00:30:00",
            account="bsmfc",
            partition="ia",
        ),
        name="POSEIDON-TRAINING",
        export="ALL",
        backend=args.backend,
    )
