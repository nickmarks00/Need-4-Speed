import math
import statistics
from typing import Tuple

import cv2

import numpy as np
from PIL import Image


class RewardHandler:
    """
    RewardHandler class

    The reward function consists currently of three parameters:
    - Velocity smoothing: reward incurred for minimisin the difference between the current and previous velocity. This is done through a buffer
    """

    def __init__(self) -> None:
        self.smoothness_buffer = Buffer(buffer_dim=2)
        self.pose_buffer = Buffer(buffer_dim=3)
        self.weights = {
            "smoothness": 0.1,
            "pose_pos": 0.01,
            "pose_theta": 0.1,
            "track": 0.1,
        }
        self.idx = 0

    def reward(
        self, l_vel: int, r_vel: int, x: float, y: float, theta: float, img: np.ndarray
    ) -> Tuple[float, float, float, float]:
        # reward_smooth = self.reward_smoothness(l_vel, r_vel)
        # reward_pose = self.reward_pose(x, y, theta)
        reward_track = self.reward_track(img)

        reward_smooth = 0
        reward_pose = 0
        # reward_track = 0
        return (
            reward_smooth + reward_pose + reward_track,
            reward_smooth,
            reward_pose,
            reward_track,
        )

    def reward_smoothness(self, l_vel: int, r_vel: int) -> float:
        self.smoothness_buffer.push([l_vel, r_vel])
        l_vel_avg = statistics.fmean(self.smoothness_buffer[0])
        r_vel_avg = statistics.fmean(self.smoothness_buffer[1])

        # Calculate the velocity differences and scale them
        l_vel_diff_scaled = (l_vel - l_vel_avg) / 10
        r_vel_diff_scaled = (r_vel - r_vel_avg) / 10

        # Use tanh to achieve a smooth transition, scale to [0, 1]
        smoothness_score = (math.tanh(-abs(l_vel_diff_scaled)) + 1) / 2 + (
            math.tanh(-abs(r_vel_diff_scaled)) + 1
        ) / 2

        # Apply the weight and normalize by 2 since two components contribute equally
        return self.weights["smoothness"] * (smoothness_score / 2)

    def reward_pose(self, x: float, y: float, theta: float) -> float:

        x_avg = statistics.fmean(self.pose_buffer[0])
        y_avg = statistics.fmean(self.pose_buffer[1])
        theta_avg = statistics.fmean(self.pose_buffer[2])

        pos_deviation = np.sqrt((x - x_avg) ** 2 + (y - y_avg) ** 2)
        theta_deviation = abs(theta - theta_avg)

        # Use tanh to smooth the reward decrease for deviations
        reward_pos = (np.tanh(-pos_deviation) + 1) / 2
        reward_theta = (np.tanh(-theta_deviation) + 1) / 2
        reward = (
            self.weights["pose_pos"] * reward_pos
            + self.weights["pose_theta"] * reward_theta
        )

        return reward

    def reward_track(self, img: np.ndarray) -> float:
        """
        Reward for keeping as much of the track in view as possible
        """
        grey_pixels = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                if np.linalg.norm(pixel) < 28 and np.max(pixel) - np.min(pixel) < 8:
                    grey_pixels += 1
        #             img[i, j] = 0
        #         else:
        #             img[i, j] = 255
        # image = Image.fromarray(img)
        # image.save(f"img_{self.idx}.png")
        # self.idx += 1
        try:
            # Ratio of grey pixels -> grey_pixels / hxw of the image
            ratio = grey_pixels / (img.shape[0] * img.shape[1])
            return self.weights["track"] * math.tanh(50 * (ratio) ** 4)
        except ZeroDivisionError:
            return 0


class Buffer:
    """
    Buffer class for different reward types
    """

    def __init__(self, buffer_dim, buffer_size=3) -> None:
        self.capacity: int = buffer_size
        self.dim: int = buffer_dim
        self.buffer = [[0] for _ in range(self.dim)]

    def push(self, vals) -> None:
        if self.is_full():
            self.pop()
        for i in range(self.dim):
            self.buffer[i].append(vals[i])

    def pop(self) -> None:
        for i in range(self.dim):
            self.buffer[i] = self.buffer[i][1:]

    def is_full(self) -> bool:
        if len(self.buffer[0]) < self.capacity:
            return False
        return True

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self) -> int:
        return len(self.buffer)

    def __setitem__(self, _, vals):
        self.push(vals)

    def __str__(self):
        return self.buffer.__str__()
