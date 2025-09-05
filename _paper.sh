# Moving to the jobs directory
cd scripts/paper/

# Generating samples
# python -u run_generation.py -m configs/models/laced-puddle-18.yml

# Computing metrics
python -u run_metrics_prior.py -m configs/models/laced-puddle-18.yml
# python -u run_metrics_posterior.py -m configs/models/laced-puddle-18.yml
# python -u run_hypoxia.py -m configs/models/laced-puddle-18.yml

# Moving back to the root directory
cd ../../

# Watching
watch squeue --me