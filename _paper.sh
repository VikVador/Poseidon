# Moving to the jobs directory
cd scripts/paper/

# Launching a jupyter notebook
python -u run_generation.py -m configs/models/laced-puddle-18.yml

# Moving back to the root directory
cd ../../

# Watching
watch squeue --me