<img  src="assets/header.gif"  />
@ Copernicus Marine Service 

<hr>
<p  align="center">
<b  style="font-size:30vw;">Understanding Ocean Deoxygenation</b>
</p>
<hr>

This repository contains the first project of my PhD thesis: a **3D generative neural network-based simulator for modeling the Black Sea's state**. The model leverages deep learning to predict physical and environmental conditions such as temperature, salinity, and currents. Ultimately, it will be able to support tasks of forecasting and data assimilation of various observations, including satellite (EO) data and float measurements.  

<hr>
<p  align="center">
<b  style="font-size:30vw;">🔱 | P O S E I D O N | 🔱</b>
</p>
<hr>

This is the name of the tool we are developping ! To set-up everything, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [poseidon](poseidon) module as a package. First, create a new Python environment, for example with [conda](https://docs.conda.io).

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
