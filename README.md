# Offline RL

This README documents how to get up and running with `d3rlpy` for running offline RL experiments.

## Setting up the environment

We will use Mamba for environment management.

### Installing Mamba 

The following instructions are geared towards a Linux system (or WSL). Note that you can also use the Micro Mamba manager if you are short on disk space. Firstly, go to the official [Github repo](https://github.com/conda-forge/miniforge#mambaforge) and download the relevant installer. Generally the one without PyPy should be sufficient. You can also use `curl` or `wget`. 

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
```

Run the install script.
```bash
bash Mambaforge-Linux-x86_64.sh
```

Accepting the defaults is fine here. Restart your shell and check for a successful install with `mamba --version`. 

### Cloning the repository 


