import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import style


class Plotter:
    def __init__(self) -> None:
        style.use("fivethirtyeight")
        self.dpi = 96
        self.reward_fig = plt.figure(
            figsize=(700 / self.dpi, 700 / self.dpi), dpi=self.dpi
        )
        self.reward_ax = self.reward_fig.add_subplot(1, 1, 1)
        self.action_fig = plt.figure()
        self.action_ax = self.action_fig.add_subplot(1, 1, 1)

    def plot_reward(self) -> None:
        reward_df = pd.read_csv(
            os.path.join(os.getcwd(), "output/log.csv"), header=None
        )

        t_steps = reward_df[0]

        self.reward_ax.plot(t_steps, reward_df[6], "c")
        self.reward_ax.plot(t_steps, reward_df[7], "r")
        self.reward_ax.plot(t_steps, reward_df[8], "g")
        self.reward_ax.plot(t_steps, reward_df[9], "b")
        self.reward_ax.legend(
            ["Total reward", "Velocity smoothing", "Pose smoothing", "Track visibility"]
        )
        self.reward_ax.set_title("Live reward")
        self.reward_ax.set_xlabel("Time steps")
        self.reward_ax.set_ylabel("Reward")
        self.reward_fig.savefig("output/reward.png", dpi=self.dpi, bbox_inches="tight")

    def plot_action_distribution(self) -> None:
        """
        Plots the distribution of actions received during operation (i.e. velocities)
        """
        df = pd.read_csv(os.path.join(os.getcwd(), "output/log.csv"), header=None)

        l_vel_arr, r_vel_arr = df[1].to_numpy(), df[2].to_numpy()
        l_vel_arr = [
            l_vel_arr[i] for i in range(len(l_vel_arr)) if abs(l_vel_arr[i]) < 100
        ]
        r_vel_arr = [
            r_vel_arr[i] for i in range(len(r_vel_arr)) if abs(r_vel_arr[i]) < 100
        ]
        self.action_ax.hist(
            l_vel_arr, np.histogram_bin_edges(l_vel_arr, bins=50)
        )  # histogram of right wheel velocities
        self.action_ax.hist(
            r_vel_arr, np.histogram_bin_edges(r_vel_arr, bins=50)
        )  # histogram of right wheel velocities

        self.action_ax.legend(["Left Wheel Velocities", "Right Wheel Velocities"])
        self.action_ax.set_title("Action Distribution")
        self.action_ax.set_xlabel("Velocity Bins")
        self.action_ax.set_ylabel("Frequency")
        self.action_fig.savefig("output/action_dist.png", bbox_inches="tight")


if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_action_distribution()
