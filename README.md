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

Clone the repository as usual.

```bash
git clone https://github.com/nickmarks00/Need-4-Speed.git
```

Now run the following to create the offline Rl branch locally and set it up to track the remote.

```bash
git branch -f offline-rl origin/offline-rl
```

---

The final step is to create the Mamba environment from file.

```bash
mamba env create --file env.yaml
```

Note that this requires you to have a system running with `cuda` installed. Otherwise it will error when it tries to install the CUDA-compiled version of `PyTorch`. If need be, I can create a separate environment file for CPU-only machines.

The new environment can now be activated.

```bash
mamba activate need4speed
```

## Usage

With the environment activated, you are now in a position to train experiments. To test everything is working well, try solving the `cartpole` experiment using a DQN.

```bash
cd demos && python3 cartpole_dqn.py
```

The training should not take long and the logs and data from the experiments should be in an appropriately named folder in `d3rly_data` and `d3rlpy_logs`. There are two other small demos you can use as well to ensure that everything is set up correctly.

### Creating an experiment with `d3rlpy`

Every experiment consists of a few basic blocks: the dataset and environment, the model and the call to train.

1. Datasets and environments are generated by importing the particular experiment of choice from `d3rlpy.datasets`. [Here](https://d3rlpy.readthedocs.io/en/latest/references/datasets.html) is a list of the available ones. Note that the `env` returned from instantiating a game with

```python
dataset, env = get_d4rl("hopper-medium-v2")
```

is an instance of an OpenGym environment and can be used in the same way consequently.

2. A model is created by importing it from the [list](https://d3rlpy.readthedocs.io/en/latest/references/algos.html) of available algorithms and configuring it as desired. If you are training with GPU, be sure to included `device=cuda:0` in the model instantiation.

3. For training, you can add loss and reward evaluators in (the latter only if you still have access to the environment during training), both of which are imported from `d3rlpy.metrics` and example configurations [here](https://d3rlpy.readthedocs.io/en/latest/tutorials/getting_started.html#setup-metrics). Then we call `fit` on the model we created. Everything from the learning rate to the number of epochs can be controlled here.

```python
iql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        save_interval=5,
        experiment_name=experiment_name,
        with_timestamp=False,
        evaluators = {
            'td_error': td_error_evaulator,
            'environment': env_evaluator
            }
)
```

Once the experiment has run, you can plot the results (here, reward) using `d3rlpy plot`:

```bash
d3rlpy plot <path to .csv file>
```

### Deploying a trained model

Once a model has been trained with `train.py`, it can be deployed to the Penguin Pi using `deploy.py`. To connect to the Penguin Pi:

1. Ensure that it is turned on. Wait until the LCD screen displays a valid WLAN address.
2. Connect to the Penguin Pi's local hotspot. The network name will be something like `penguinpi:xx:xx:xx` where `x` represents some hexadecimal value. The password is `egb439123`.
3. Run `python3 deploy.py`.
4. If you have connection issues, ensure that the IP address set in `deploy.py` parser arguments matches that displayed on the Pi's LCD screen, and the port is correctly set to `8080`. You can also test the connection by opening a browser and navigating to `http://192.168.50.1:8080` - it should present you with a control interface for the Penguin Pi.
