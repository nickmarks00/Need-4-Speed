import numpy as np
import cv2
import os
import shutil
import sys
import csv
import pygame  # python package for GUI

from utils.penguin_pi import PenguinPi  # access the robot
from utils.rewards import RewardHandler
from utils.plotter import Plotter
from utils.globals import DIMS


class Operate:
    def __init__(self, args):
        self.folder = "output/"
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.mkdir(self.folder)
        os.mkdir(os.path.join(self.folder, "images"))

        # initialise data parameters
        self.pibot = PenguinPi(args.ip, args.port)

        self.command = {
            "motion": [0, 0],
        }
        self.quit = False
        self.w = DIMS["width"]
        self.h = DIMS["height"]
        self.c = DIMS["channels"]
        self.image_id = 0
        self.img = np.zeros([self.h, self.w, self.c], dtype=np.uint8)

        self.rewards = RewardHandler()
        self.plotter = Plotter()

    # wheel control
    def control(self):
        self.pibot.set_velocity(self.command["motion"], tick=20, turning_tick=5)

    # camera control
    def take_pic(self):
        f_ = os.path.join(self.folder, "images", f"img_{self.image_id}.png")
        image = self.pibot.get_image()
        self.img = image[240 - self.h :, :, :]  # crop to 120x320x3
        cv2.imwrite(f_, image)
        self.image_id += 1

    # keyboard teleoperation
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command["motion"][0] = min(self.command["motion"][0] + 1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command["motion"][0] = max(self.command["motion"][0] - 1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command["motion"][1] = min(self.command["motion"][1] + 1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command["motion"][1] = max(self.command["motion"][1] - 1, -1)
            # stop
            elif event.type == pygame.KEYUP:
                self.command["motion"] = [0, 0]
            # quit
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            print("\nExiting program...")
            self.pibot.stop()
            sys.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar="", type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar="", type=int, default=8080)
    args, _ = parser.parse_known_args()

    canvas = pygame.display.set_mode((700, 700))

    operate = Operate(args)

    reset = operate.pibot.resetEncoder()

    motor_speeds = []

    path_to_reward = os.path.join(os.getcwd(), "output/reward.png")
    t_steps = 0

    """
    Operate.csv format
    | Time steps | Left vel | Right vel | Total reward | Vel smooth | Pose smooth | Track vis |
    """

    while True:
        try:
            with open(os.path.join(operate.folder, "log.csv"), "a") as f:
                writer = csv.writer(f, delimiter=",")
                operate.update_keyboard()
                operate.control()
                if operate.command["motion"] != [0, 0]:  # actual  input given
                    t_steps += 1
                    operate.take_pic()
                    l_vel, r_vel = operate.pibot.getEncoders()
                    x, y, theta = operate.pibot.get_pose()
                    rewards = operate.rewards.reward(
                        l_vel, r_vel, x, y, theta, operate.img
                    )
                    vals = (t_steps, l_vel, r_vel, *rewards)
                    writer.writerow(vals)
                    f.flush()
                    if t_steps > 1:
                        operate.plotter.plot_reward()
                        bg = pygame.image.load(path_to_reward)
                        canvas.blit(bg, (0, 0))
                reset = operate.pibot.resetEncoder()
                pygame.display.update()
        except KeyboardInterrupt:
            print("\nExiting program...")
            operate.pibot.stop()
            sys.exit()
