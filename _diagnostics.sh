# Moving to the jobs directory
cd experiments/diagnostics/

# Launching a jupyter notebook
python -u run_diagnostics.py -c configs/diagnostics.yml -m sweet-forest-13 -v last -cpt all -ts all -s test -g -cm

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me