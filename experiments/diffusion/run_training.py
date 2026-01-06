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
    config_cluster = configs[0].get("Cluster").copy()

    # Renaming clear variables names into SLURM-compatible ones
    nodes         = config_cluster.get("nodes", 1)
    gpus_per_node = config_cluster.get("gpus-per-node", config_cluster.get("gpus"))
    cpus_per_node = config_cluster.get("cpus-per-node", config_cluster.get("cpus"))
    ram_per_node  = config_cluster.get("ram-per-node",  config_cluster.get("ram"))

    config_cluster_slurm = {
        "account":       config_cluster.get("account"),
        "partition":     config_cluster.get("partition"),
        "time":          config_cluster.get("time"),
        "nodes":         nodes,
        "gpus-per-node": gpus_per_node,
        "cpus-per-task": cpus_per_node,
        "mem":           ram_per_node,
    }

    # Local
    if args.backend == "async":
        training(
            **configs[0].get("Training"),
            config_cluster=config_cluster,
        )

    # Cluster
    else:
        # Build torchrun command for DDP
        if nodes > 1:
            # Multi-node: requires rendezvous
            interpreter = (
                f"torchrun --nnodes {nodes} --nproc-per-node {gpus_per_node} "
                f"--rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 "
                f"--rdzv_id=$SLURM_JOB_ID"
            )
        else:
            # Single-node: standalone mode
            interpreter = f"torchrun --nnodes 1 --nproc-per-node {gpus_per_node} --standalone"

        @job(array=len(configs), **config_cluster_slurm)
        def BS_train(i: int):
            training(
                **configs[i].get("Training"),
                config_cluster=config_cluster,
            )

        schedule(
            BS_train,
            name="POSEIDON-TRAINING",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
