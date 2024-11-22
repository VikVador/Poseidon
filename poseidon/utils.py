r"""Poseidon - Random tools to help throughout the project."""

import psutil
import torch

# isort: split


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
