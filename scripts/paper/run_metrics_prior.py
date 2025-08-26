r"""Script to launch a diagnostics pipeline."""

import argparse
import wandb

from dawgz import job, schedule

# isort: split
from poseidon.diagnostics.const import DATES_PRIOR
from poseidon.diagnostics.metrics import computing_metrics_prior
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
        load_configuration("configs/partitions/cpu_prior.yml")[0],
        load_configuration(args.model)[0],
    )

    # Creates an ID for the run
    config_setup["wandb_id"] = wandb.util.generate_id()

    # ==========
    #   PRIOR
    # ==========
    @job(array=24, **config_cluster_cpu)
    def prior_metrics(i: int):
        computing_metrics_prior(
            date=DATES_PRIOR[i],
            config= {
                "p(x)_samples": 512,
                **config_model,
            }
        )

    schedule(
        prior_metrics,
        name="POSEIDON-DIAGNOSTICS-PRIOR",
        backend="slurm",
        export="ALL",
    )
