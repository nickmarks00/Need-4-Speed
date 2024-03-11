from d3rlpy.algos import DQNConfig
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.metrics import TDErrorEvaluator

import torch


if __name__ == "__main__":
    dataset, env = get_cartpole()

    device = "cuda:0" if torch.cuda.is_available() else None

    # if you don't use GPU, set device=None instead.
    dqn = DQNConfig().create(device=device)

    # initialize neural networks with the given observation shape and action size.
    # this is not necessary when you directly call fit or fit_online method.
    dqn.build_with_dataset(dataset)

    env_evaluator = EnvironmentEvaluator(
        env
    )  # evaluates the reward if environment still accessible
    rewards = env_evaluator(dqn, dataset=None)
    td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

    # Train the model
    dqn.fit(
        dataset,
        n_steps=10000,
        evaluators={
            "td_error": td_error_evaluator,
            "environment": env_evaluator,
        },
        n_steps_per_epoch=500,
        save_interval=5,
        experiment_name="cartpole_dqn",
    )
