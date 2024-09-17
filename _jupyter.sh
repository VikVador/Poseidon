# Moving to the jobs directory
cd cluster/jobs

# Launching a jupyter notebook
sbatch jupyter.sbatch

# Moving back to the root directory
cd ../../

# Watching the job status
watch squeue --me
