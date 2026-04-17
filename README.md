<img  src="assets/header.gif"  />
@ Copernicus Marine Service 

<hr>
<p  align="center">
<b  style="font-size:1vw;">The Physical Limit of Neural Hypoxia Detection in the Black Sea from Satellite Observations</b>
</p>
<hr>

Coastal hypoxia (O2 < 63 [mmol/m^3]) threatens ocean health worldwide. Bottom oxygen consumed by respiration cannot be renewed, making monitoring essential to protect vulnerable marine ecosystems and reduce biodiversity loss. Despite the growing availability of Black Sea satellite observations, no operational system currently exploits them to directly infer the oxygen state in real time. This can be framed as a Bayesian inverse problem relating surface observations to the complete Black Sea states. Here, we solve it using a deep generative neural network trained on numerical model outputs, providing a tractable approximation of the true posterior distribution of sea states. We find that accurate state estimation is limited to the mixing layer, because its homogeneity makes surface conditions representative of subsurface states. During summer, we detect 38\% of all hypoxic events shelf-wide with a precision of 47\%. Improving results will likely require longer assimilation windows or additional observations. 

<hr>
<p  align="center">
<b  style="font-size:1vw;">Installation</b>
</p>
<hr>

To set-up everything, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [poseidon](poseidon) module as a package. First, create a new Python environment, for example with [conda](https://docs.conda.io).

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
