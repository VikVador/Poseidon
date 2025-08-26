r"""Script to launch a diagnostics pipeline."""

import argparse
import wandb

from dawgz import job, schedule

# isort: split
from poseidon.diagnostics.const import DATES_POSTERIOR
from poseidon.diagnostics.metrics import computing_metrics_posterior
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

    config_setup, config_cluster_cpu, config_model = (
        load_configuration("configs/setup.yml")[0],
        load_configuration("configs/partitions/cpu_posterior.yml")[0],
        load_configuration(args.model)[0],
    )

    # Creates an ID for the run
    config_setup["wandb_id"] = wandb.util.generate_id()

    # ==========
    #   PRIOR
    # ==========
    @job(array=24, **config_cluster_cpu)
    def posterior_metrics(i: int):
        computing_metrics_posterior(
            date=DATES_POSTERIOR[i],
            config=config_model,
        )

    schedule(
        posterior_metrics,
        name="POSEIDON-DIAGNOSTICS-POSTERIOR",
        backend="slurm",
        export="ALL",
    )
