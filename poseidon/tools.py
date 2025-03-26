r"""A collection of (random) tools designed to help throughout the project."""

import psutil
import torch

from typing import Dict, Sequence


def wandb_get_hyperparameter_score(configs: Sequence[Dict]) -> Dict[str, float]:
    r"""Computes an importance score for each hyperparameter.

    Information:
        By logging these scores on Weights & Biases, it is easily possible to
        use the 'parameter importance' feature to identify the most important
        hyperparameters that influence the model's performance.

    Arguments:
        configs: A sequence of configuration dictionaries containing hyperparameters.

    Returns:
        A dictionary mapping each hyperparameter to its importance score.
    """

    scores = {}

    for cfg in configs:
        for k, v in cfg.items():
            if k == "batch_size":
                scores["Batch Size"] = v

            elif k == "blanket_neighbors":
                scores["Blanket Size (K)"] = v * 2 + 1

            elif k == "steps_gradient_accumulation":
                scores["Gradient Accumulation Steps"] = v

            elif k == "learning_rate":
                scores["Learning Rate (init. value)"] = v

            elif k == "weight_decay":
                scores["Weight Decay"] = v

            elif k == "kernel_size":
                scores["Kernel Size"] = v

            elif k == "mod_features":
                scores["Modulation Features"] = v

            elif k == "ffn_scaling":
                scores["Feed-Forward Network Scaling"] = v

            elif k == "hid_channels":
                scores["Number of Stages"] = len(v)
                for i, h in enumerate(v):
                    scores[f"Hidden Channels (Stage {i})"] = h

            elif k == "hid_blocks":
                for i, b in enumerate(v):
                    scores[f"Hidden Blocks (Stage {i})"] = b

            elif k == "dropout":
                scores["Dropout"] = v

            elif k == "attention_heads":
                for l in range(scores["Number of Stages"]):
                    scores[f"Attention Heads (Stage {l})"] = 0 if str(l) not in v else 1

            elif k == "features":
                scores["Mesh Encoding Size"] = v

            elif k == "n_layers":
                scores["Number of Layers (Siren)"] = v

    return scores


def memory_usage() -> None:
    r"""Display memory usage for CPU and GPU in [GB]."""

    # Conversion factor
    to_gb = 1 / (1024**3)

    # -- CPU --
    cpu_memory_used = round(psutil.virtual_memory().used * to_gb, 2)

    # -- GPU --
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory = {
            "total": round(gpu_props.total_memory * to_gb, 2),
            "reserved": round(torch.cuda.memory_reserved(0) * to_gb, 2),
            "allocated": round(torch.cuda.memory_allocated(0) * to_gb, 2),
        }
    else:
        gpu_memory = {"total": 0, "reserved": 0, "allocated": 0}

    memory_stats = {
        "CPU_Used_GB": cpu_memory_used,
        "GPU_Total_GB": gpu_memory["total"],
        "GPU_Reserved_GB": gpu_memory["reserved"],
        "GPU_Allocated_GB": gpu_memory["allocated"],
    }

    print(
        f"Memory [GB] | CPU - Used: {memory_stats['CPU_Used_GB']} | "
        f"GPU - Total: {memory_stats['GPU_Total_GB']} - "
        f"Reserved: {memory_stats['GPU_Reserved_GB']} - "
        f"Allocated: {memory_stats['GPU_Allocated_GB']}"
    )
