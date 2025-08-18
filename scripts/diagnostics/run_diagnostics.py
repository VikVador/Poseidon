r"""Script to launch a diagnostics pipeline."""

import argparse
import os
import wandb

from dawgz import after, job, schedule

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.diagnostics.generate import generate_conditional, generate_unconditional
from poseidon.diagnostics.vizualize import (
    plot_conditional_distributions,
    plot_reconstructions,
    plot_unconditional,
    plot_unconditional_distributions,
)
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
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        help="GPU partition on which launch generations.",
        default="gpu",
        choices=["gpu", "ia"],
    )

    #
    # fmt: off
    args = parser.parse_args()

    config_setup, config_cluster_cpu, config_cluster_gpu, config_model, config_sampling_prior, config_sampling_posterior = (
        load_configuration("configs/setup.yml")[0],
        load_configuration("configs/partitions/cpu.yml")[0],
        load_configuration("configs/partitions/gpu.yml")[0] if args.partition == "gpu" else load_configuration("configs/partitions/ia.yml")[0],
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
    assert config_sampling_prior["nb_nowcasts"] >= 32, "ERROR - The number of nowcasts to generate must be at least 32."

    path_folder_prior = PATH_MODEL / config_model["model"] / "nowcasts" / "unconditional"
    if not os.path.exists(path_folder_prior):
        os.makedirs(path_folder_prior)

    @job(array=config_sampling_prior["nb_nowcasts"], **config_cluster_gpu)
    def prior_generate(i: int):
        generate_unconditional(
            index=i,
            config= {
                **config_model,
                **config_sampling_prior,
            }
        )

    @after(prior_generate)
    @job(**config_cluster_cpu)
    def prior_d():
        plot_unconditional_distributions(
            config=config_model,
            config_setup=config_setup,
        )

    @after(prior_d)
    @job(**config_cluster_cpu)
    def prior_v():
        plot_unconditional(
            config=config_model,
            config_setup=config_setup,
        )

    @after(prior_v)
    @job(**config_cluster_cpu)
    def prior_r():
        plot_reconstructions(
            config=config_model,
            config_setup=config_setup,
        )

    schedule(
        prior_r,
        name="POSEIDON-DIAGNOSTICS-PRIOR",
        backend="slurm",
        export="ALL",
    )

    # ==============
    #   POSTERIOR
    # ==============
    path_folder_posterior = PATH_MODEL / config_model["model"] / "nowcasts" / "conditional"
    if not os.path.exists(path_folder_posterior):
        os.makedirs(path_folder_posterior)

    @job(array=12, **config_cluster_gpu)
    def post_generate(i: int):
        generate_conditional(
            index=i,
            config= {
                **config_model,
                **config_sampling_posterior,
            }
        )

    @after(post_generate)
    @job(**config_cluster_cpu)
    def post_d():
        plot_conditional_distributions(
            config=config_model,
            config_setup=config_setup,
        )

    schedule(
        post_d,
        name="POSEIDON-DIAGNOSTICS-POSTERIOR",
        backend="slurm",
        export="ALL",
    )
