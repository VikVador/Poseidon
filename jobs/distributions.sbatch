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
# A Submission script to compute the distributions of the data
#
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=DISTRIBUTION
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=medium
#SBATCH --output=../logs/distribution.log
#SBATCH --error=../logs/distribution_error.log
#SBATCH --mail-user=vmangeleer@uliege.be
#SBATCH --mail-type=FAIL
#
# ---------------
# Loading modules
# ---------------
#module purge

# ----
# Main
# ----
cd ../src/debs/

# Running the script
echo "Running the script"
python -u __distributions.py