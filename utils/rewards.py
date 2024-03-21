import numpy as np
import math
from typing import Tuple


class RewardHandler:
    """
    RewardHandler class

    The reward function consists currently of three parameters:
    - Velocity smoothing: reward incurred for minimisin the difference between the current and previous velocity. This is done through a buffer
    """

    def __init__(self) -> None:
        self.smoothness_buffer = Buffer(2)
        self.pose_buffer = Buffer(3)
        self.weights = {
            "smoothness": 0.01,
            "pose_pos": 0.05,
            "pose_theta": 0.1,
            "track": 0.5,
        }

    def reward(
        self, l_vel: int, r_vel: int, x: float, y: float, theta: float, img: np.ndarray
    ) -> Tuple[float, float, float, float]:
        reward_smooth = self.reward_smoothness(l_vel, r_vel)
        reward_pose = self.reward_pose(x, y, theta)
        reward_track = self.reward_track(img)
        return (
            reward_smooth + reward_pose + reward_track,
            reward_smooth,
            reward_pose,
            reward_track,
        )

    def reward_smoothness(self, l_vel: int, r_vel: int) -> float:
        self.smoothness_buffer.push([l_vel, r_vel])
        l_vel_avg = np.mean(self.smoothness_buffer[0])
        r_vel_avg = np.mean(self.smoothness_buffer[1])

        try:
            reward = self.weights["smoothness"] * (
                1 / math.sqrt(abs(l_vel - l_vel_avg))
                + 1 / math.sqrt(abs(r_vel - r_vel_avg))
            )
        except ZeroDivisionError:
            if math.sqrt((l_vel_avg - l_vel) ** 2 + (r_vel_avg - r_vel) ** 2 < 1e-2):
                reward = 0.1
            else:
                reward = 0

        return reward

    def reward_pose(self, x: float, y: float, theta: float) -> float:
        x_avg = np.mean(self.pose_buffer[0])
        y_avg = np.mean(self.pose_buffer[1])
        theta_avg = np.mean(self.pose_buffer[2])

        reward = self.weights["pose_pos"] * (
            math.exp(-1 * ((x - x_avg) ** 2 + (y - y_avg) ** 2))
        ) + self.weights["pose_theta"] * math.exp(-1 * (theta - theta_avg) ** 2)
        return reward

    def reward_track(self, img: np.ndarray) -> float:
        """
        Reward for keeping as much of the track in view as possible
        """
        grey_pixels = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pixel = img[i, j]
                if np.linalg.norm(pixel) < 150 and np.max(pixel) - np.min(pixel) < 10:
                    grey_pixels += 1

        try:
            return (
                self.weights["track"]
                * math.log(grey_pixels)
                / (img.shape[0] * img.shape[1])
            )
        except ZeroDivisionError:
            return 0


class Buffer:
    """
    Buffer class for different reward types
    """

    def __init__(self, buffer_dim, buffer_size=3) -> None:
        self.capacity: int = buffer_size
        self.dim: int = buffer_dim
        self.buffer: np.ndarray = np.array([[0] for _ in range(self.dim)])

    def push(self, vals) -> None:
        if self.is_full():
            self.pop()
        for i in range(self.dim):
            np.append(self.buffer[i], vals[i])

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
