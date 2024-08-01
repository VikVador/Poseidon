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
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=JUPYTER
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=2
#SBATCH --mem=120G
#SBATCH --partition=ia
#SBATCH --output=../logs/jupiter_notebook.log
#SBATCH --error=../logs/jupiter_notebook.err
#SBATCH --mail-user=vmangeleer@uliege.be
#SBATCH --mail-type=FAIL
#
# ----
# Main
# ----
# Removing all the modules previously loaded
module purge

# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/ESA---Black-Sea-Deoxygenation/ --no-browser --port 8888