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
#     /!\   Don't forget to activate your conda environment (esa) before running this script   /!\
#
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=JUPYTER
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=gpu
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
# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/ESA---Black-Sea-Deoxygenation/ --no-browser --port 8888