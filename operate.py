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
        self.img = self.pibot.get_image()

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, "images", f"img_{self.image_id}.png")
        image = self.pibot.get_image()
        image = image[240 - self.h :, :, :]  # crop to 120x320x3
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

    # with open(os.path.join(operate.folder, "operate.csv"), "w") as f:

    while True:
        try:
            f = open(os.path.join(operate.folder, "operate.csv"), "a")
            writer = csv.writer(f, delimiter=",")
            operate.update_keyboard()
            operate.control()
            if operate.command["motion"] != [0, 0]:  # actual  input given
                operate.take_pic()
                operate.save_image()
                l_vel, r_vel = operate.pibot.getEncoders()
                x, y, theta = operate.pibot.get_pose()
                rewards = operate.rewards.reward(l_vel, r_vel, x, y, theta, operate.img)
                # if motor_speeds is not None:
                vals = (l_vel, r_vel, *rewards)
                writer.writerow(vals)
                f.close()
                operate.plotter.plot_reward()
            reset = operate.pibot.resetEncoder()
        except KeyboardInterrupt:
            print("\nExiting program...")
            operate.pibot.stop()
            sys.exit()
