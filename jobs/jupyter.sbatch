#!/bin/bash
# -------------------------------------------------------
#
#        |
#       / \
#      / _ \                  ESA - PROJECT
#     |.o '.|
#     |'._.'|          BLACK SEA DEOXYGENATION EMULATOR
#     |     |
#   ,'|  |  |`.             BY VICTOR MANGELEER
#  /  |  |  |  \
#  |,-'--|--'-.|                2023-2024
#
#
# -------------------------------------------------------
#
# Documentation
# -------------
# A Submission script for Lucia to activate a notebook !
#
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=JUPYTER
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32000
#SBATCH --gres="gpu:1"
#SBATCH --partition=medium
#SBATCH --output=../logs/jupiter_notebook.log
#SBATCH --mail-user=vmangeleer@uliege.be
#SBATCH --mail-type=FAIL
#
# ---------------
# Loading modules
# ---------------
module purge

# ----
# Main
# ----
# Activating the project environement
conda activate esa

# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/ESA---Black-Sea-Deoxygenation/ --no-browser --port 8888
