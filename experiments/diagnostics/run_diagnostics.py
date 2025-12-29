r"""Script to launch a diagnostics pipeline."""

import argparse
import os
import random
import wandb

from dawgz import after, job, schedule

# isort: split
from poseidon.config import PATH_MODEL
from poseidon.diagnostics import (
    DIAGNOSTICS_DATES_EXPERIMENTS,
    DIAGNOSTICS_DATES_POSTERIOR,
    DIAGNOSTICS_DATES_PRIOR,
    HYPOXIA_THRESHOLDS,
)
from poseidon.diagnostics.generate import (
    generate_from_observations,
    generate_from_posterior,
    generate_from_prior,
    generate_reconstructions,
)
from poseidon.diagnostics.metrics import (
    compute_distance,
    compute_hypoxia_classification,
    compute_power_spectra_density,
    compute_spread_skill,
)
from poseidon.diagnostics.visualize import (
    visualize_denoiser,
    visualize_distance,
    visualize_ensemble_prior,
    visualize_spread_skill_ratio,
)
from poseidon.training.tools import load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a diagnostics pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the diagnostics .yml configuration file.",
    )

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
        "--component",
        "-cpt",
        type=str,
        default="all",
        choices=["denoiser", "posterior", "experiments", "all"],
    )

    parser.add_argument(
        "--timespan",
        "-ts",
        type=str,
        default="reduced",
        choices=["reduced", "all"],
    )

    parser.add_argument(
        "--generate",
        "-g",
        action='store_true',
    )

    parser.add_argument(
        "--compute_metrics",
        "-cm",
        action='store_true',
    )

    args              = parser.parse_args()
    config_generation = load_configuration(args.config)[0]

    # Security
    path_model = PATH_MODEL / args.model
    if os.path.exists(path_model) is False:
        print("ERROR - Model {} does not exist.".format(args.model))
        exit()

    # Used to subsample dates for generation
    factor = 4 if args.timespan == "reduced" else 1
    DIAGNOSTICS_DATES_PRIOR       = DIAGNOSTICS_DATES_PRIOR[::factor]
    DIAGNOSTICS_DATES_POSTERIOR   = DIAGNOSTICS_DATES_POSTERIOR[::factor]
    DIAGNOSTICS_DATES_EXPERIMENTS = DIAGNOSTICS_DATES_EXPERIMENTS[::factor]

    # Extracting configurations
    config_wandb, config_noise, config_sampling_prior, config_sampling_posterior, config_cluster = (
        config_generation["wandb"],
        config_generation["Noise"],
        config_generation["Prior"],
        config_generation["Posterior"],
        config_generation["Cluster"],
    )

    # Determine if we need to run visualizations
    needs_visualization = args.compute_metrics

    # Extending wandb configuration
    config_wandb["resume"] = "allow"
    config_wandb["name"] = args.model
    config_wandb["id"] = wandb.util.generate_id()

    # Set wandb offline if no visualizations are needed
    if not needs_visualization:
        config_wandb["mode"] = "offline"

    # Stores what needs to be scheduled
    jobs_queue = []

    # ==========
    # GENERATION
    # ==========
    # Determines wether or not to create arrays for generation
    gen_prior       = True if args.component in ["denoiser",    "all"] else False
    gen_posterior   = True if args.component in ["posterior",   "all"] else False
    gen_experiments = True if args.component in ["experiments", "all"] else False

    array_size_prior       = len(DIAGNOSTICS_DATES_PRIOR)       if args.generate and gen_prior       else 1
    array_size_posterior   = len(DIAGNOSTICS_DATES_POSTERIOR)   if args.generate and gen_posterior   else 1
    array_size_experiments = len(DIAGNOSTICS_DATES_EXPERIMENTS) if args.generate and gen_experiments else 1

    # Creating folders to store results
    path_folder_prior = path_model / "generation" / "prior"
    if not os.path.exists(path_folder_prior):
        os.makedirs(path_folder_prior)

    path_folder_posterior = path_model/ "generation" / "posterior"
    if not os.path.exists(path_folder_posterior):
        os.makedirs(path_folder_posterior)

    path_folder_reconstruction = path_model / "denoising"
    if not os.path.exists(path_folder_reconstruction):
        os.makedirs(path_folder_reconstruction)

    # Creating jobs
    @job(array=1, account = config_cluster["account"], **config_cluster["sampling_reconstruction"])
    def GEN_REC(i: int) -> None:
        generate_reconstructions(
            config = {
                "model": args.model,
                "best": args.version,
                **config_noise,
                **config_sampling_prior,
            }
        ) if gen_prior else print("Nothing to generate for prior (reconstructions).")

    @job(array=array_size_prior, account = config_cluster["account"], **config_cluster["sampling_prior"],)
    def GEN_PRI(i: int) -> None:
        generate_from_prior(
            date = DIAGNOSTICS_DATES_PRIOR[i],
            config = {
                "model": args.model,
                "best": args.version,
                **config_noise,
                **config_sampling_posterior,
            }
        ) if gen_prior else print("Nothing to generate for prior (visualizations).")

    @job(array=array_size_posterior, account = config_cluster["account"], **config_cluster["sampling_posterior"])
    def GEN_POS(i: int) -> None:
        generate_from_posterior(
            date = DIAGNOSTICS_DATES_POSTERIOR[i],
            config = {
                "model": args.model,
                "best": args.version,
                **config_noise,
                **config_sampling_posterior,
            }
        ) if gen_posterior else print("Nothing to generate for posterior.")

    @job(array=array_size_experiments, account = config_cluster["account"], **config_cluster["sampling_posterior"])
    def GEN_EXP(i: int) -> None:
        generate_from_observations(
            date = DIAGNOSTICS_DATES_EXPERIMENTS[i],
            config = {
                "model": args.model,
                "best": args.version,
                **config_noise,
                **config_sampling_posterior,
            }
        ) if gen_experiments else print("Nothing to generate for experiments.")

    # Adding jobs
    jobs_queue += [GEN_REC, GEN_EXP, GEN_PRI, GEN_POS]

    # ===================
    # ANALYSIS | DENOISER
    # ===================
    if args.component == "denoiser" or args.component == "all":

        array_size_distances = len(DIAGNOSTICS_DATES_PRIOR) if args.compute_metrics else 1

        path_folder_distance = path_model / "diagnostics" / "distance"
        if not os.path.exists(path_folder_distance):
            os.makedirs(path_folder_distance)

        @after(GEN_PRI)
        @job(array=array_size_distances, account = config_cluster["account"], **config_cluster["computing_metrics_heavy"])
        def COM_DIS(i: int) -> None:
            compute_distance(
                date = DIAGNOSTICS_DATES_PRIOR[i],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_prior else print("Nothing to compute for distance.")

        @after(GEN_PRI)
        @job(array=array_size_distances, account = config_cluster["account"], **config_cluster["computing_metrics_medium"])
        def COM_PSD(i: int) -> None:
            compute_power_spectra_density(
                date = DIAGNOSTICS_DATES_PRIOR[i],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_prior else print("Nothing to compute for distance.")

        # Queueing jobs
        jobs_queue += [COM_DIS, COM_PSD]

        # Only add visualization jobs if we need them
        if needs_visualization and gen_prior:

            @after(GEN_PRI)
            @job(array=1, account = config_cluster["account"], **config_cluster["visualizations"])
            def VIS_PRI(i: int) -> None:
                visualize_ensemble_prior(
                    date = random.choice(DIAGNOSTICS_DATES_PRIOR),
                    config = {"model": args.model},
                    config_wandb=config_wandb
                )

            @after(GEN_REC)
            @job(array=1, account = config_cluster["account"], **config_cluster["visualizations"])
            def VIS_DEN(i: int) -> None:
                visualize_denoiser(
                    config={"model": args.model},
                    config_wandb=config_wandb
                )

            @after(COM_DIS)
            @job(array=1, account = config_cluster["account"], **config_cluster["visualizations"])
            def VIS_DIS(i: int) -> None:
                visualize_distance(
                    dates=DIAGNOSTICS_DATES_PRIOR,
                    config={"model": args.model},
                    config_wandb=config_wandb
                )

            jobs_queue += [VIS_PRI, VIS_DIS, VIS_DEN]

    # ===================
    # ANALYSIS | COMPLETE
    # ===================
    if args.component == "posterior" or args.component == "all":

        path_folder_spread_skill = path_model / "diagnostics" / "spread_skill"
        if not os.path.exists(path_folder_spread_skill):
            os.makedirs(path_folder_spread_skill)

        path_folder_classification = path_model / "diagnostics" / "classification"
        if not os.path.exists(path_folder_classification):
            os.makedirs(path_folder_classification)

        # Creating pairs of dates and thresholds (in 4 to handle array size limitations)
        index_quarter = len(HYPOXIA_THRESHOLDS) // 4

        pairs_date_threshold_1, nb_thresh_1 = [(str(d), float(v))
                for d in DIAGNOSTICS_DATES_POSTERIOR
                for v in HYPOXIA_THRESHOLDS[:index_quarter]], len(HYPOXIA_THRESHOLDS[:index_quarter])

        pairs_date_threshold_2, nb_thresh_2 = [(str(d), float(v))
                for d in DIAGNOSTICS_DATES_POSTERIOR
                for v in HYPOXIA_THRESHOLDS[index_quarter:2*index_quarter]], len(HYPOXIA_THRESHOLDS[index_quarter:2*index_quarter])

        pairs_date_threshold_3, nb_thresh_3 = [(str(d), float(v))
                for d in DIAGNOSTICS_DATES_POSTERIOR
                for v in HYPOXIA_THRESHOLDS[2*index_quarter:3*index_quarter]], len(HYPOXIA_THRESHOLDS[2*index_quarter:3*index_quarter])

        pairs_date_threshold_4, nb_thresh_4 = [(str(d), float(v))
                for d in DIAGNOSTICS_DATES_POSTERIOR
                for v in HYPOXIA_THRESHOLDS[3*index_quarter:]], len(HYPOXIA_THRESHOLDS[3*index_quarter:])

        array_size_spread_skill     = len(DIAGNOSTICS_DATES_POSTERIOR)                    if args.compute_metrics else 1
        array_size_classification_1 = int(len(DIAGNOSTICS_DATES_POSTERIOR) * nb_thresh_1) if args.compute_metrics else 1
        array_size_classification_2 = int(len(DIAGNOSTICS_DATES_POSTERIOR) * nb_thresh_2) if args.compute_metrics else 1
        array_size_classification_3 = int(len(DIAGNOSTICS_DATES_POSTERIOR) * nb_thresh_3) if args.compute_metrics else 1
        array_size_classification_4 = int(len(DIAGNOSTICS_DATES_POSTERIOR) * nb_thresh_4) if args.compute_metrics else 1

        @after(GEN_POS)
        @job(array=array_size_spread_skill, account = config_cluster["account"], **config_cluster["computing_metrics_medium"])
        def COM_SSR(i: int) -> None:
            compute_spread_skill(
                date = DIAGNOSTICS_DATES_POSTERIOR[i],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_posterior else print("Nothing to compute for spread-skill.")

        @after(GEN_POS)
        @job(array=array_size_classification_1, account = config_cluster["account"], **config_cluster["computing_metrics_light"])
        def COM_CLA1(i: int) -> None:
            compute_hypoxia_classification(
                date = pairs_date_threshold_1[i][0],
                threshold = pairs_date_threshold_1[i][1],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_posterior else print("Nothing to compute for hypoxia classification (1).")

        @after(GEN_POS)
        @job(array=array_size_classification_2, account = config_cluster["account"], **config_cluster["computing_metrics_light"])
        def COM_CLA2(i: int) -> None:
            compute_hypoxia_classification(
                date = pairs_date_threshold_2[i][0],
                threshold = pairs_date_threshold_2[i][1],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_posterior else print("Nothing to compute for hypoxia classification (2).")

        @after(GEN_POS)
        @job(array=array_size_classification_3, account = config_cluster["account"], **config_cluster["computing_metrics_light"])
        def COM_CLA3(i: int) -> None:
            compute_hypoxia_classification(
                date = pairs_date_threshold_3[i][0],
                threshold = pairs_date_threshold_3[i][1],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_posterior else print("Nothing to compute for hypoxia classification (3).")

        @after(GEN_POS)
        @job(array=array_size_classification_4, account = config_cluster["account"], **config_cluster["computing_metrics_light"])
        def COM_CLA4(i: int) -> None:
            compute_hypoxia_classification(
                date = pairs_date_threshold_4[i][0],
                threshold = pairs_date_threshold_4[i][1],
                config = {"model": args.model}
            ) if args.compute_metrics and gen_posterior else print("Nothing to compute for hypoxia classification (4).")

        # Queueing jobs
        jobs_queue += [COM_SSR, COM_CLA1, COM_CLA2, COM_CLA3, COM_CLA4]

        # Only add visualization jobs if we need them
        if needs_visualization and gen_posterior:
            @after(COM_SSR)
            @job(array=1, account = config_cluster["account"], **config_cluster["visualizations"])
            def VIS_SSR(i: int) -> None:
                visualize_spread_skill_ratio(
                    dates=DIAGNOSTICS_DATES_POSTERIOR,
                    config={"model": args.model},
                    config_wandb=config_wandb
                )

            jobs_queue += [VIS_SSR]

    # ===================
    # ANALYSIS |  ERROR
    # ===================
    if args.component not in ["denoiser", "posterior", "experiments", "all"]:
        print("ERROR - Unknown analysis option.")
        exit()

    # Launching jobs
    schedule(*jobs_queue, name="Poseidon-Diagnostics", backend="slurm", export="ALL")
