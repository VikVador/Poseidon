<img  src="assets/header.gif"  />
<hr>
<p  align="center">
<b  style="font-size:30vw;">Understanding Ocean Deoxygenation</b>
</p>
<hr>

This repository contains code for developing a 3D neural network-based simulator of the Black Sea's state. The simulator is designed to model and predict various physical and environmental conditions of the Black Sea, such as temperature, salinity, and currents, using deep learning techniques. The goal is to provide accurate, data-driven insights into the Black Sea's dynamics, enabling better understanding and forecasting of its state over time.

<hr>
<p  align="center">
<b  style="font-size:30vw;">Poseidon</b>
</p>
<hr>

This is the name of the tool we are developping ! To set-up everything, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [posiedon](poseidon) module as a package. First, create a new Python environment, for example with [conda](https://docs.conda.io).

```
conda create -n poseidon python=3.11
conda activate poseidon
```

Then, install the [poseidon](poseidon) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cu121
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yml
```
