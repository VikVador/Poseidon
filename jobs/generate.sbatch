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
# A Submission script to preprocess the data
#
# --------------
# Job parameters
# --------------
#SBATCH --account=bsmfc
#SBATCH --job-name=GENERATE_DATA
#SBATCH --time=08:00:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=240G
#SBATCH --partition=medium
#SBATCH --output=../logs/generate_data.log
#SBATCH --error=../logs/generate_data_error.log
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
python -u generate.py
