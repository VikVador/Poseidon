r"""Script to launch a diagnostics pipeline."""

import argparse
import wandb

from dawgz import job, schedule

# isort: split
from poseidon.diagnostics.hypoxia import determine_threshold
from poseidon.training.parser import load_configuration

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
        load_configuration("configs/partitions/cpu.yml")[0],
        load_configuration(args.model)[0],
    )

    # Creates an ID for the run
    config_setup["wandb_id"] = wandb.util.generate_id()

    # ==========
    #   PRIOR
    # ==========
    @job(array=1, **config_cluster)
    def find_hypoxia(i: int):
        determine_threshold()

    schedule(
        find_hypoxia,
        name="POSEIDON-DIAGNOSTICS-PRIOR",
        backend="slurm",
        export="ALL",
    )
