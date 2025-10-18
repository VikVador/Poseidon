# Moving to the jobs directory
cd experiments/diagnostics/

# Launching a jupyter notebook
python -u run_diagnostics.py --model magic-dew-12     --version last --generate --compute_metrics --component both --timespan all
python -u run_diagnostics.py --model swift-morning-11 --version last --generate --compute_metrics --component both --timespan all

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me