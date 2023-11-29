#!/bin/bash
# --------------------------------------
#             ESA - PROJECT
# --------------------------------------
# @ Victor Mangeleer
#
# Submission script for Lucia
#
#
#SBATCH --account=bsmfc
#
#SBATCH --job-name=JUPYTER
#SBATCH --time=12:00:00 # hh:mm:ss
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000 # megabytes
#SBATCH --gres="gpu:0"
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
# Code
# ----
# Activating the project environement (to be done before hand)
# micromamba activate esa

# Creating an instance of the jupyter notebook
jupyter lab --ip='*' --notebook-dir=~/esa/ --no-browser --port 8888
