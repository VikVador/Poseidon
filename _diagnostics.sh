# Moving to the jobs directory
cd experiments/diagnostics/

# Launching diagnostics
python -u run_diagnostics.py -c configs/diagnostics-test.yml -m fancy-wind-40 -v last -cpt all -ts all -s test -g -cm 

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me