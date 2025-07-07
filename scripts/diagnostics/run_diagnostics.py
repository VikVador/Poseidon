r"""Script to launch a diagnostics pipeline."""

import argparse

from dawgz import after, job, schedule

# isort: split
from poseidon.diagnostics.generate import generate_unconditional
from poseidon.diagnostics.vizualize import plot_unconditional
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

    # Security
    assert config_sampling_prior["nb_nowcasts"] >= 24, "ERROR - The number of nowcasts to generate must be at least 24."

    # Unconditionnal nowcast generation configuration
    @job(array=config_sampling_prior["nb_nowcasts"], **config_cluster_gpu)
    def generate(i: int):
        generate_unconditional(
            index=i,
            config= {
                **config_model,
                **config_sampling_prior,
            }
        )

    # Diagnostics of unconditionnal nowcasts
    @after(generate)
    @job(**config_cluster_cpu)
    def prior():
        plot_unconditional(
            config=config_model,
            config_setup=config_setup,
        )


    schedule(
        prior,
        name="POSEIDON-DIAGNOSTICS",
        backend="slurm",
        export="ALL",
    )
