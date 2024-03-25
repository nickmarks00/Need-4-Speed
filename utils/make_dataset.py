# Package imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import glob
from tqdm import tqdm

# Local imports
from utils.rewards import RewardHandler
from utils.globals import DIMS


class DatasetFactory:
    def __init__(self, path_to_dataset: str) -> None:
        self.path_to_dataset = path_to_dataset
        self.w = DIMS["width"]
        self.h = DIMS["height"]
        self.c = DIMS["channels"]
        self.create()

    def create(self) -> None:
        print("Creating dataset from:", self.path_to_dataset)

        df = pd.read_csv(os.path.join(self.path_to_dataset, "log.csv"), header=None)
        num_rows = len(df)
        print("Number of rows in dataset: ", num_rows)
        print("Number of columns in dataset: ", len(df.columns))

        print("\nReading images...")
        self.get_observations()

        print("\nReading actions and rewards...")
        self.actions: np.ndarray = df[[1, 2]].to_numpy()
        self.normalise_actions(axis=0)

        # ======= IMPORTANT ======= #
        print("Checking if rewards are present...")
        if len(df.columns) == 6:
            print("\nNo rewards found. Calculating rewards...")
            self.calculate_reward()
        # ======= IMPORTANT ======= #
        else:
            print("\nRewards found. Reading rewards...")
            self.rewards: np.ndarray = df[2].to_numpy()

        # Safe to process images now that rewards are calculated
        self.image_process()

        self.terminals = np.zeros(num_rows)
        self.terminals[-1] = 1
        print("Actions shape:", self.actions.shape)
        print("Rewards shape:", self.rewards.shape)
        print("Terminals shape:", self.terminals.shape)

        try:
            assert self.actions.max() <= 1 and self.actions.min() >= -1
        except AssertionError:
            print("Actions not properly normalised")

        print("Dataset created!")

    def get_observations(self) -> None:
        images = []
        for filename in glob.glob(
            os.path.join(self.path_to_dataset, "images", "*.png")
        ):
            img = Image.open(filename)
            img = np.array(img, dtype=np.uint8)
            images.append(img)

        self.observations = np.array(images)

    def image_process(self) -> None:

        self.observations = self.observations.transpose(0, 3, 2, 1)

        try:
            assert self.observations.shape[1:] == (self.c, self.w, self.h)
            print(f"\nSuccessfully read {len(self.observations)} images")
            print(f"Observations shape: {self.observations.shape}")
        except AssertionError:
            print(
                f"Expected images of shape {(self.c, self.w, self.h)}, got {self.observations.shape[1:]}"
            )

    def calculate_reward(self) -> None:
        """
        Calculates the reward offline. It will also write the reward values to file to cache output
        """
        csv_path = os.path.join(self.path_to_dataset, "log.csv")
        df = pd.read_csv(csv_path, header=None)

        print("Found {} rows in dataset".format(len(df)))

        reward_vec = []
        handler = RewardHandler()

        for i, (action, img) in tqdm(
            enumerate(zip(self.actions, self.observations)), desc="Processing rewards"
        ):
            x, y, theta = df[3][i], df[4][i], df[5][i]
            rewards = handler.reward(action[0], action[1], x, y, theta, img)
            reward_vec.append(rewards)

        reward_vec = pd.DataFrame(reward_vec)
        self.rewards = reward_vec[0].to_numpy()  # first column is the sum reward
        df = pd.concat([df, reward_vec], axis=1)
        df.to_csv(csv_path, header=False, index=False)

    # Continuous action space must be between [-1, 1]
    def normalise_actions(self, axis=None) -> None:
        amax = self.actions.max(axis)
        amin = self.actions.min(axis)
        abs_max = np.where(-amin > amax, amin, amax)

        self.actions = self.actions / abs_max
