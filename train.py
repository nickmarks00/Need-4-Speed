import numpy as np
from d3rlpy.algos import IQLConfig
from d3rlpy.dataset import MDPDataset, FrameStackTransitionPicker
import torch

# from d3rlpy.metrics import TDErrorEvaluator, EnvironmentEvaluator

from make_dataset import DatasetFactory


PATH_TO_DATASET = "/home/chumbers/Downloads/"


def main() -> None:
    experiment_name = "iql_pibot_mock"

    device = "cuda:0" if torch.cuda.is_available() else None

    raw_dataset = DatasetFactory(PATH_TO_DATASET)

    dataset = MDPDataset(
        observations=raw_dataset.observations,
        actions=np.random.random((100, 2)),
        rewards=raw_dataset.rewards,
        terminals=raw_dataset.terminals,
        transition_picker=FrameStackTransitionPicker(n_frames=4),
    )

    iql = IQLConfig().create(device=device)
    iql.build_with_dataset(dataset)

    # Train the model
    iql.fit(
        dataset,
        n_steps=1000,
        n_steps_per_epoch=100,
        save_interval=10,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
