from typing import List
import math


class RewardHandler:
    """
    RewardHandler class

    The reward function consists currently of three parameters:
    - Velocity smoothing: reward incurred for minimisin the difference between the current and previous velocity. This is done through a buffer
    """

    def __init__(self) -> None:
        self.time_since_infraction = 0
        self.buffer = Buffer()
        self.smoothness_weight = 0.01

    def calculate_reward(self) -> float:
        reward = math.sqrt(self.time_since_infraction)
        self.time_since_infraction += 1
        return reward

    def handle_infraction(self) -> int:
        self.time_since_infraction = 0
        return 0

    def reward_smoothness(self, l_vel: float, r_vel: float) -> float:
        self.buffer.push([l_vel, r_vel])
        l_vel_avg = sum(self.buffer[0]) / len(self.buffer[0])
        r_vel_avg = sum(self.buffer[1]) / len(self.buffer[0])

        try:
            reward = self.smoothness_weight * (
                1 / math.sqrt(abs(l_vel - l_vel_avg))
                + 1 / math.sqrt(abs(r_vel - r_vel_avg))
            )
        except ZeroDivisionError:
            if math.sqrt((l_vel_avg - l_vel) ** 2 + (r_vel_avg - r_vel) ** 2 < 1e-2):
                reward = 0.1
            else:
                reward = 0

        return reward


class Buffer:
    """
    Buffer class for velocity smoothing
    """

    def __init__(self, buffer_size=3) -> None:
        self.capacity: int = buffer_size
        self.buffer: List[List[float]] = [[0], [0]]

    def push(self, vals: List[float]) -> None:
        if not self.is_full():
            self.buffer[0].append(vals[0])  # left velocity
            self.buffer[1].append(vals[1])  # right velocity
        else:
            self.pop()
            self.buffer[0].append(vals[0])  # left velocity
            self.buffer[1].append(vals[1])  # right velocity
        print(self.buffer)

    def pop(self) -> None:
        self.buffer[0] = self.buffer[0][1:]
        self.buffer[1] = self.buffer[1][1:]

    def is_full(self):
        if len(self.buffer[0]) < self.capacity:
            return False
        return True

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def __setitem__(self, _, val: List[float]):
        self.push(val)
