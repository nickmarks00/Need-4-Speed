import numpy as np
import pandas as pd
from PIL import Image
import glob


# Note that channel dimension HAS to come first according to the docs
IMAGE_DIM = (3, 320, 240)  # channel x width x height


class DatasetFactory:
    def __init__(self, path_to_dataset: str) -> None:
        self.path_to_dataset = path_to_dataset
        self.create()

    def create(self) -> None:
        print("Creating dataset from:", self.path_to_dataset)

        df = pd.read_csv(self.path_to_dataset + "/actions1.csv", header=None)
        num_rows = len(df)
        print("Number of rows in dataset:", num_rows)

        self.actions: np.ndarray = df[[0, 1]].to_numpy()
        self.normalise_actions(axis=0)
        # self.rewards: np.ndarray = df[2].to_numpy()
        self.rewards = np.random.random(num_rows)
        self.terminals = np.random.randint(2, size=num_rows)
        print("Actions shape:", self.actions.shape)
        print("Rewards shape:", self.rewards.shape)

        try:
            assert self.actions.max() <= 1 and self.actions.min() >= -1
        except AssertionError:
            print("Actions not properly normalised")

        self.get_observations()

        print("Dataset created!")

    def get_observations(self) -> None:
        images = []
        for filename in glob.glob(self.path_to_dataset + "/pibot_dataset/*.png"):
            img = Image.open(filename)
            # img = img.resize((320, 240))
            img = np.array(img, dtype=np.uint8)
            images.append(img)

        self.observations = np.array(images).transpose(0, 3, 2, 1)

        try:
            assert self.observations.shape[1:] == IMAGE_DIM
            print(f"Observations shape: {self.observations.shape}")
        except AssertionError:
            print(
                f"Expected images of shape {IMAGE_DIM}, got {self.observations.shape[1:]}"
            )

    # Continuous action space must be between [-1, 1]
    def normalise_actions(self, axis=None) -> None:
        amax = self.actions.max(axis)
        amin = self.actions.min(axis)
        abs_max = np.where(-amin > amax, amin, amax)

        self.actions = self.actions / abs_max
