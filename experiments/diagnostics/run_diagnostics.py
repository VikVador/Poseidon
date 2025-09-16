r"""Script to launch a diagnostics pipeline."""

import argparse
import os

from dawgz import job, schedule

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.diagnostics import DIAGNOSTICS_DATES_POSTERIOR, DIAGNOSTICS_DATES_PRIOR
from poseidon.diagnostics.generate import generate_from_posterior, generate_from_prior
from poseidon.training.tools import load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a diagnostics pipeline.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="last",
        choices=["best", "last"],
    )

    parser.add_argument(
        "--analysis",
        "-a",
        type=str,
        default="partial",
        choices=["partial", "complete"],
    )

    parser.add_argument(
        "--generate",
        "-g",
        action='store_true',
    )

    args              = parser.parse_args()
    config_generation = load_configuration("configs/generation.yml")[0]

    # Security
    path_model = PATH_MODEL / args.model
    if os.path.exists(path_model) is False:
        print("ERROR - Model {} does not exist.".format(args.model))
        exit()

    # Used to subsample dates for generation
    factor = 4 if args.analysis == "partial" else 1
    DIAGNOSTICS_DATES_PRIOR     = DIAGNOSTICS_DATES_PRIOR[::factor]
    DIAGNOSTICS_DATES_POSTERIOR = DIAGNOSTICS_DATES_POSTERIOR[::factor]

    # ==========
    # GENERATION
    # ==========
    if args.generate:

        # Extracting configurations
        config_noise, config_sampling_prior, config_sampling_posterior, config_cluster = (
            config_generation["Noise"],
            config_generation["Prior"],
            config_generation["Posterior"],
            config_generation["Cluster"],
        )

        # Setting up folders
        path_folder_prior = path_model / "generation" / "prior"
        if not os.path.exists(path_folder_prior):
            os.makedirs(path_folder_prior)

        path_folder_posterior = path_model/ "generation" / "posterior"
        if not os.path.exists(path_folder_posterior):
            os.makedirs(path_folder_posterior)

        @job(array=len(DIAGNOSTICS_DATES_PRIOR), account = config_cluster["account"], **config_cluster["sampling_prior"],)
        def GEN_PRI(i: int):
            generate_from_prior(
                date = DIAGNOSTICS_DATES_PRIOR[i],
                config = {
                    "model": args.model,
                    "best": args.version,
                    **config_noise,
                    **config_sampling_posterior,
                }
            )

        @job(array=len(DIAGNOSTICS_DATES_POSTERIOR), account = config_cluster["account"], **config_cluster["sampling_posterior"])
        def GEN_POS(i: int):
            generate_from_posterior(
                date = DIAGNOSTICS_DATES_POSTERIOR[i],
                config = {
                    "model": args.model,
                    "best": args.version,
                    **config_noise,
                    **config_sampling_prior,
                }
            )

        # Launching jobs
        schedule(GEN_PRI, name="Poseidon-Generation-Prior",     backend="slurm", export="ALL")
        schedule(GEN_POS, name="Poseidon-Generation-Posterior", backend="slurm", export="ALL")

    # ========
    # ANALYSIS
    # ========
