r"""A collection of tools designed to help throughout the project."""

import psutil
import torch


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
