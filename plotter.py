"""
Creates plots for the training of a given model. You will need to specify as a runtime argument the path to the log folder, e.g. demos/d3rlpy_logs/<model-name>_<code>
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot(dir: str):

    # Load the enviroment data AKA the reward
    env_df = pd.read_csv(dir + "/environment.csv", header=None)

    # Load the loss data
    loss_df = pd.read_csv(dir + "/loss.csv", header=None)

    # Plot the loss data
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(env_df[0], env_df[2])
    ax1.set_title("Reward")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reward")
    ax2.plot(loss_df[0], loss_df[2])
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    f.savefig(dir + "/results.png")


if __name__ == "__main__":

    dir = sys.argv[1]
    plot(dir)