# Moving to the jobs directory
cd scripts/diagnostics/

# Launching a jupyter notebook
python -u run_diagnostics.py -m configs/models/laced-puddle-18.yml

# Moving back to the root directory
cd ../../

# Watching
watch squeue --me