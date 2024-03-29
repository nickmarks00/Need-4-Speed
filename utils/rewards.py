import numpy as np
import math
import statistics
from typing import Tuple


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

    def reward(
        self, l_vel: int, r_vel: int, x: float, y: float, theta: float, img: np.ndarray
    ) -> Tuple[float, float, float, float]:
        # reward_smooth = self.reward_smoothness(l_vel, r_vel)
        # reward_pose = self.reward_pose(x, y, theta)
        reward_smooth = 0
        reward_pose = 0
        reward_track = self.reward_track(img)
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

        # Use tanh to achieve a smooth transition.
        smoothness_score = (math.tanh(-abs(l_vel_diff_scaled)) + 1) / 2 + (
            math.tanh(-abs(r_vel_diff_scaled)) + 1
        ) / 2

        # Apply the weight. You might adjust this formula based on your specific needs.
        return self.weights["smoothness"] * smoothness_score

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
                if np.linalg.norm(pixel) < 150 and np.max(pixel) - np.min(pixel) < 10:
                    grey_pixels += 1
        try:
            ratio = grey_pixels / (img.shape[0] * img.shape[1] * img.shape[2])
            return self.weights["track"] * math.exp(100 * (ratio) ** 3)
        except ZeroDivisionError:
            return 0

    """
    def reward_track(self, img: np.ndarray) -> float:
            
            Calculates the reward for keeping as much of the track in view as possible,
            
            # Define thresholds 
            norm_threshold = 150
            color_diff_threshold = 10
            gray_patch_size_threshold = 50  
            
            # Calculate norms and color differences
            norms = np.linalg.norm(img, axis=2)
            color_diffs = np.max(img, axis=2) - np.min(img, axis=2)
            
            # Create a binary mask for gray pixels based on the criteria
            gray_pixels_mask = (norms < norm_threshold) & (color_diffs < color_diff_threshold)
            binary_mask = np.uint8(gray_pixels_mask) * 255

            # Perform connected components analysis to filter out small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            # Count pixels in components larger than the threshold
            large_components_mask = np.isin(labels, np.where(stats[:, cv2.CC_STAT_AREA] > gray_patch_size_threshold)[0])
            grey_pixels = np.sum(large_components_mask)
            
                ratio = grey_pixels / (img.shape[0] * img.shape[1])
                return self.weights["track"] * math.exp(100 * (ratio) ** 3)
            except ZeroDivisionError:
                return 0
    """


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
