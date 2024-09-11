#!/bin/bash
# -------------------------------------------------------------
#
#        |
#       / \                 ESA - MITHO PROJECT
#      / _ \
#     |.o '.|      "GENERATIVE MODELS FOR HYPOXIA FORECASTING"
#     |'._.'|
#     |     |               by VICTOR MANGELEER
#   ,'|  |  |`.
#  /  |  |  |  \                2023-2024
#  |,-'--|--'-.|
#
# --------------------------------------------------------------
# https://eo4society.esa.int/projects/mitho/
#
# Moving to the jobs directory
cd cluster/jobs

# Launching a jupyter notebook
sbatch jupyter.sbatch

# Moving back to the root directory
cd ../../

# Watching the job status
watch squeue --me
