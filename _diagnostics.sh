# Moving to the jobs directory
cd experiments/diagnostics/

# Launching a jupyter notebook
python -u run_diagnostics.py --model laced-puddle-18 --version last --generate --compute_metrics --component both --timespan all 

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me