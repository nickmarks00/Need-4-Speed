"""
Solves Hopper game with Implicit Q-learning (IQL) algorithm.
"""

from d3rlpy.datasets import get_atari
from d3rlpy.algos import CQLConfig
from d3rlpy.metrics import TDErrorEvaluator, EnvironmentEvaluator
import os

import torch

if __name__ == "__main__":
    experiment_name = "cql_breakout"

    device = "cuda:0" if torch.cuda.is_available() else None

    try:
        os.system(f"rm -rf d3rlpy_logs/{experiment_name}")
    except:
        pass

    # List of available tasks is avaiable here: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
    dataset, env = get_atari("breakout-mixed-v2")

    # Using the vanilla config settings and placing model on GPU
    # Refer to https://d3rlpy.readthedocs.io/en/latest/references/algos.html#d3rlpy.algos.cql.IQL for more details on parameters
    cql = CQLConfig().create(device="cuda:0")

    cql.build_with_dataset(dataset)

    td_error_evaulator = TDErrorEvaluator(episodes=dataset.episodes)
    env_evaluator = EnvironmentEvaluator(env)
    rewards = env_evaluator(cql, dataset=None)

    # Train the model
    cql.fit(
        dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        save_interval=5,
        experiment_name=experiment_name,
        with_timestamp=False,
        evaluators={"td_error": td_error_evaulator, "environment": env_evaluator},
    )

    os.system(f"d3rlpy plot d3rlpy_logs/{experiment_name}/environment.csv")
