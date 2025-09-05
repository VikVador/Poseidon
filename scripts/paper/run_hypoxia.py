r"""Script to launch a diagnostics pipeline."""

import argparse
import numpy as np
import wandb

from dawgz import job, schedule

# isort: split
from poseidon.diagnostics.hypoxia import evaluate_hypoxia_threshold
from poseidon.training.tools import load_configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a validation pipeline.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to the model .yml configuration file.",
    )

    #
    # fmt: off
    args = parser.parse_args()

    config_setup, config_cluster, config_model = (
        load_configuration("configs/setup.yml")[0],
        load_configuration("configs/partitions/cpu_hypoxia.yml")[0],
        load_configuration(args.model)[0],
    )

    # Creates an ID for the run
    config_setup["wandb_id"] = wandb.util.generate_id()

    # Determine number of experiments to do:
    nb_experiments = 21

    # ==========
    #   PRIOR
    # ==========
    @job(array=nb_experiments, **config_cluster)
    def find_hypoxia(i: int):

        # Commuting tresholds
        tresholds = np.linspace(0, 200, nb_experiments)

        evaluate_hypoxia_threshold(
            hypoxia_bias=float(tresholds[i]),
            experiment_index=i,
        )

    schedule(
        find_hypoxia,
        name="POSEIDON-DIAGNOSTICS-PRIOR",
        backend="slurm",
        export="ALL",
    )
