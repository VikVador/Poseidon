r"""Script to launch a training pipeline."""

import argparse

from dawgz import job, schedule

# isort: split
from poseidon.training.tools import load_configuration
from poseidon.training.training import training

# fmt: off
#
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

    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_cluster = configs[0].get("Cluster")
    nb_gpus        = config_cluster.get("gpus")
    batch_size     = configs[0].get("Training").get("config_dataloader").get("batch_size")

    # Security
    assert (nb_gpus <= batch_size), f"ERROR - For // training, batch size ({batch_size}) > ({nb_gpus})."

    # Local
    if args.backend == "async":
        training(
            **configs[0].get("Training"),
            config_cluster=config_cluster,
        )

    # Cluster
    else:
        @job(array=len(configs), **config_cluster)
        def BS_train(i: int):
            training(
                **configs[i].get("Training"),
                config_cluster=config_cluster,
            )

        schedule(
            BS_train,
            name="POSEIDON-TRAINING",
            backend="slurm",
            export="ALL",
        )
