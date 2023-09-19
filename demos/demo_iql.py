"""
Solves Hopper game with Implicit Q-learning (IQL) algorithm.
"""

from d3rlpy.datasets import get_d4rl
from d3rlpy.algos import IQLConfig
from d3rlpy.metrics import TDErrorEvaluator, EnvironmentEvaluator
import os

if __name__ == "__main__":
    experiment_name = "iql_hopper_medium_v2"

    os.system(f'rm -rf d3rlpy_logs/{experiment_name}')

    # List of available tasks is avaiable here: https://github.com/Farama-Foundation/d4rl/wiki/Tasks
    dataset, env = get_d4rl("hopper-medium-v2") 

    # Using the vanilla config settings and placing model on GPU
    # Refer to https://d3rlpy.readthedocs.io/en/latest/references/algos.html#d3rlpy.algos.iql.IQL for more details on parameters
    iql = IQLConfig().create(device="cuda:0")

    iql.build_with_dataset(dataset)

    td_error_evaulator = TDErrorEvaluator(episodes=dataset.episodes)
    env_evaluator = EnvironmentEvaluator(env)
    rewards = env_evaluator(iql, dataset=None)

    # Train the model 
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

    os.system(f'd3rlpy plot d3rlpy_logs/{experiment_name}/environment.csv')
