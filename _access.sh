# Prints the link to the jupyter notebook
grep "http://localhost:8888/lab?token=" cluster/logs/jupiter_notebook.log | awk '{print $NF}'

# Print the ssh command to connect to the jupyter notebook
node=$(squeue --me | awk 'NR>1 {print $8}')
printf "ssh -fNL 8888:%s:8888 lucia\n" "$node"