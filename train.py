import os
import numpy as np
from d3rlpy.algos import IQLConfig
from d3rlpy.dataset import MDPDataset, FrameStackTransitionPicker
import torch

from utils.make_dataset import DatasetFactory
from utils.file_handler import next_path


PATH_TO_DATASET = "/home/chumbers/Downloads/"


def main() -> None:
    experiment_name = "iql"

    device = "cuda:0" if torch.cuda.is_available() else None

    raw_dataset = DatasetFactory(PATH_TO_DATASET)

    dataset = MDPDataset(
        observations=raw_dataset.observations,
        actions=np.random.random((100, 2)),
        rewards=raw_dataset.rewards,
        terminals=raw_dataset.terminals,
        # transition_picker=FrameStackTransitionPicker(n_frames=4),
    )

    iql = IQLConfig().create(device=device)
    iql.build_with_dataset(dataset)

    # Train the model
    iql.fit(
        dataset,
        n_steps=20,
        n_steps_per_epoch=5,
        save_interval=2,
        experiment_name=experiment_name,
    )

    model_path = next_path("models/iql-%s.d3")
    iql.save(model_path)


if __name__ == "__main__":
    main()
