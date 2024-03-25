# Package imports
import os
from d3rlpy.algos import IQLConfig
from d3rlpy.dataset import MDPDataset
import torch

# Local imports
from utils.make_dataset import DatasetFactory
from utils.file_handler import next_path


def main() -> None:

    path_to_dataset = os.path.join(os.getcwd(), "output")

    experiment_name = "iql"

    device = "cuda:0" if torch.cuda.is_available() else None

    raw_dataset = DatasetFactory(path_to_dataset)

    dataset = MDPDataset(
        observations=raw_dataset.observations,
        actions=raw_dataset.actions,
        rewards=raw_dataset.rewards,
        terminals=raw_dataset.terminals,
        # transition_picker=FrameStackTransitionPicker(n_frames=4),
    )

    iql = IQLConfig().create(device=device)
    iql.build_with_dataset(dataset)

    print("\n=========================================")
    print("Training IQL...")
    print("=========================================")
    # Train the model

    epochs = 250
    n_steps_per_epoch = 250
    iql.fit(
        dataset,
        n_steps=n_steps_per_epoch * epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        save_interval=25,
        experiment_name=experiment_name,
    )

    # check if models/ directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        print("\nCreated models/ directory")

    model_path = next_path("models/iql-%s.d3")
    iql.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
