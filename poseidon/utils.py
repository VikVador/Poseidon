r"""Poseidon - Random tools to help throughout the project."""

import ast
import psutil
import torch

from typing import Dict, Sequence

# isort: split
from poseidon.config import POSEIDON_GRID, POSEIDON_PTRC, PROJECT_FOLDER_DATA


def generate_paths() -> Dict[tuple, Sequence[str]]:
    r"""Generate paths to physical and biogeochemical datasets (1980 to 2022).

    Returns:
        A dictionary which, given a key "YEAR-MONTH", returns paths to fetch data.
    """
    with open(POSEIDON_GRID, "r") as file:
        physics_data = ast.literal_eval(file.read())
    with open(POSEIDON_PTRC, "r") as file:
        biogeochemistry_data = ast.literal_eval(file.read())

    # Creating the complete dictionnary of path mappings to data
    dataset_paths = {}
    for date_key in physics_data:
        combined_paths = physics_data[date_key] + biogeochemistry_data[date_key]
        dataset_paths[date_key] = [str(PROJECT_FOLDER_DATA) + path for path in combined_paths]
    return dataset_paths


def MemoryUsage() -> None:
    r"""A helper function to display memory usage of the CPU & GPU."""
    to_GB = 1 / (1024**3)
    CPU_memory = psutil.virtual_memory().used * to_GB
    GPU_memory_total = torch.cuda.get_device_properties(0).total_memory * to_GB
    GPU_memory_reserved = torch.cuda.memory_reserved(0) * to_GB
    GPU_memory_allocated = torch.cuda.memory_allocated(0) * to_GB
    CPU_memory = round(CPU_memory, 2)
    GPU_memory_total = round(GPU_memory_total, 2)
    GPU_memory_reserved = round(GPU_memory_reserved, 2)
    GPU_memory_allocated = round(GPU_memory_allocated, 2)
    print(
        f"RAM[GB] | CPU - Used: {CPU_memory} | GPU - Total: {GPU_memory_total} - Reserved: {GPU_memory_reserved} - Allocated: {GPU_memory_allocated}"
    )
