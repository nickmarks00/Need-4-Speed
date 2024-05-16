# Package imports
import sys
import time

import d3rlpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.file_handler import fetch_model_path
from utils.globals import DIMS

# Local imports
from utils.penguin_pi import PenguinPi  # access the robot
from utils.rewards import RewardHandler


class Operate:
    def __init__(self, params):
        self.pibot = PenguinPi(params.ip, params.port)
        self.device = "cuda:0" if torch.cuda.is_available() else None

        self.w = DIMS["width"]
        self.h = DIMS["height"]
        self.c = DIMS["channels"]
        self.img = np.zeros(
            [1, self.c, self.w, self.h], dtype=np.uint8
        )  # this is the format that P-Pi images are received as

        self.path_to_model = fetch_model_path()
        self.model = d3rlpy.load_learnable(self.path_to_model, device=self.device)
        self.steps = [1]
        self.speeds = np.zeros((0, 2))
        self.preds = np.zeros((0, 2))
        self.rewards = np.zeros((0, 1))
        self.reward_mod = RewardHandler()

    def take_pic(self):
        """
        Takes picture via penguin pi USB cam
        """
        img = self.pibot.get_image()
        self.img = img[240 - self.h :, :, :]
        self.img = np.array(self.img, dtype=np.uint8).transpose(2, 1, 0)
        self.img = np.expand_dims(self.img, axis=0)

    def control(self):
        """
        Iteratively uses recent image to generate prediction of wheel speeds
        """
        predictions = self.model.predict(self.img)
        predictions = np.array(predictions[0])
        drive = [0, 0]
        for idx, prediction in enumerate(predictions):
            drive[idx] = int(((40 * prediction) + 7) / 1.3) + 10

        self.speeds = np.vstack([self.speeds, np.array(drive)])
        self.preds = np.vstack([self.preds, predictions])
        _ = self.pibot.predict_velocity(drive[0], drive[1])

    def update_plot(self, ax1, ax2, ax1_a, ax2_a, ax3):
        """
        Plots predicted vs. actual wheel velocities.
        """

        ax1.clear()
        ax2.clear()
        ax1_a.clear()
        ax2_a.clear()
        ax3.clear()

        ax1.plot(
            self.steps, self.speeds[:, 0], label="Left Wheel Velocity", color="blue"
        )
        ax1_a.plot(
            self.steps,
            self.preds[:, 0],
            label="Left Wheel Velocity",
            color="blue",
            linestyle="--",
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Velocity (Real)")
        # ax1_a.set_ylabel("Velocity (Pred)")
        ax1.legend(loc="upper left")

        ax2.plot(
            self.steps, self.speeds[:, 1], label="Left Wheel Velocity", color="red"
        )
        ax2_a.plot(
            self.steps,
            self.preds[:, 1],
            label="Left Wheel Velocity",
            color="red",
            linestyle="--",
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (Real)")
        # ax2_a.set_ylabel("Velocity (Pred)")
        ax2.legend(loc="upper left")

        x, y, theta = self.pibot.get_pose()
        rewards = self.reward_mod.reward(
            self.speeds[-1][0], self.speeds[-1][1], x, y, theta, self.img
        )
        self.rewards = np.vstack([self.rewards, np.array(rewards[0])])
        ax3.plot(self.steps, self.rewards, label="Total reward", color="green")
        ax3.legend(loc="upper left")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Reward")

        self.steps.append(self.steps[-1] + 1)
        plt.pause(0.01)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar="", type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar="", type=int, default=8080)
    args, _ = parser.parse_known_args()

    START = False
    operate = Operate(args)
    prompt = input(f"\nDo you wish to deploy model {operate.path_to_model}? (y/n)... ")
    if prompt == "y":
        START = True
    else:
        print("Exiting...")
        sys.exit()

    RESET = operate.pibot.resetEncoder()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1_a = ax1.twinx()
    ax2_a = ax2.twinx()
    fig.suptitle("Actual vs. Predicted Wheel Velocities")

    while START:
        try:
            operate.take_pic()
            operate.control()
            operate.update_plot(ax1, ax2, ax1_a, ax2_a, ax3)
            time.sleep(0.1)
            RESET = operate.pibot.resetEncoder()
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            print("\nZeroing velocities...")
            operate.pibot.stop()
            sys.exit()
