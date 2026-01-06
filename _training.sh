# Moving to the jobs directory
cd experiments/diffusion/

# Launching a jupyter notebook
python -u run_training.py --backend slurm --config configs/unet.yml 

# Moving back to the root directory
cd ../../

# Observing the queue
watch squeue --me