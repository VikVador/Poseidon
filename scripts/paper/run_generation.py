r"""Script to launch a diagnostics pipeline."""

import argparse
import os
import wandb

from dawgz import job, schedule

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.diagnostics.const import DATES_PRIOR
from poseidon.diagnostics.generate import generate_unconditional
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

    config_setup, config_cluster_gpu, config_cluster_ia, config_model, config_sampling_prior, config_sampling_posterior = (
        load_configuration("configs/setup.yml")[0],
        load_configuration("configs/partitions/gpu.yml")[0],
        load_configuration("configs/partitions/ia.yml")[0],
        load_configuration(args.model)[0],
        load_configuration("configs/sampling/prior.yml")[0],
        load_configuration("configs/sampling/posterior.yml")[0],
    )

    # Creates an ID for the run
    config_setup["wandb_id"] = wandb.util.generate_id()

    # ==========
    #   PRIOR
    # ==========
    # Security
    path_folder_prior = PATH_MODEL / config_model["model"] / "nowcasts" / "unconditional"
    if not os.path.exists(path_folder_prior):
        os.makedirs(path_folder_prior)

    @job(array=24, **config_cluster_gpu)
    def prior_generate(i: int):
        generate_unconditional(
            index=i,
            date=DATES_PRIOR[i],
            config= {
                **config_model,
                **config_sampling_prior,
            }
        )

    schedule(
        prior_generate,
        name="POSEIDON-DIAGNOSTICS-PRIOR",
        backend="slurm",
        export="ALL",
    )

    # ==============
    #   POSTERIOR
    # ==============
    # path_folder_posterior = PATH_MODEL / config_model["model"] / "nowcasts" / "conditional"
    # if not os.path.exists(path_folder_posterior):
    #     os.makedirs(path_folder_posterior)

    # @job(array=24, **config_cluster_ia)
    # def post_generate(i: int):
    #     generate_conditional(
    #         index=i,
    #         config= {
    #             **config_model,
    #             **config_sampling_posterior,
    #         }
    #     )

    # schedule(
    #     post_generate,
    #     name="POSEIDON-DIAGNOSTICS-POSTERIOR",
    #     backend="slurm",
    #     export="ALL",
    # )
