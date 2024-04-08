# Package imports
import os
from d3rlpy.algos import CQLConfig #IQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import (
    StandardObservationScaler,
    MinMaxActionScaler,
    StandardRewardScaler,
)
import torch

# Local imports
from utils.make_dataset import DatasetFactory
from utils.file_handler import next_path


def main() -> None:

    path_to_dataset = os.path.join(os.getcwd(), "output")

    device = "cuda:0" if torch.cuda.is_available() else None

    raw_dataset = DatasetFactory(path_to_dataset)

    dataset = MDPDataset(
        observations=raw_dataset.observations,
        actions=raw_dataset.actions,
        rewards=raw_dataset.rewards,
        terminals=raw_dataset.terminals,
        # transition_picker=FrameStackTransitionPicker(n_frames=4),
    )

    # Pass scalers so d3rlpy normalises data
    observation_scaler = StandardObservationScaler()
    action_scaler = MinMaxActionScaler()
    reward_scaler = StandardRewardScaler()

    """
    iql = IQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    ).create(device=device)
    iql.build_with_dataset(dataset)
    """

    cql = CQLConfig(
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    ).create(device=device)
    cql.build_with_dataset(dataset)

    print("\n=========================================")
    print("Training IQL...")
    print("=========================================\n")

    epochs = 50
    n_steps_per_epoch = 250
    experiment_name = f"iql-straight-{epochs}epochs-{n_steps_per_epoch}steps"
    """
    iql.fit(
        dataset,
        n_steps=n_steps_per_epoch * epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        save_interval=25,
        experiment_name=experiment_name,
    )
    """
    cql.fit(
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
    # policy_path = next_path("models/iql_sp-%s.pt")
    # p_model_path = next_path("models/iql_sm-%s.pt")
    # iql.save(model_path)
    cql.save(model_path)
    # iql.save_policy(policy_path)
    # iql.save_model(p_model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
