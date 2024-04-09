import csv
import os
import shutil
import sys

import cv2
import numpy as np
import pygame  # python package for GUI
from utils.globals import DIMS

from utils.penguin_pi import PenguinPi  # access the robot
from utils.plotter import Plotter
from utils.rewards import RewardHandler


class Operate:
    def __init__(self, args):
        self.mode = "one_shot" if args.one_shot else "bare"
        self.csv_mode = "w" if args.one_shot else "a"
        self.folder = "output/"
        if os.path.exists(self.folder) and args.one_shot:
            confirm = input(
                "ONE SHOT MODE: You are about to overwrite the output folder. Continue? (y/n): "
            )
            if confirm == "y":
                print("Deleting existing output folder...")
                shutil.rmtree(self.folder)

                print("Attaching reward handler and live plotter...")
                self.rewards = RewardHandler()
                self.plotter = Plotter()
                self.reward_plot = os.path.join(os.getcwd(), "output/reward.png")
            else:
                sys.exit()

        if not os.path.exists(self.folder):
            print("Creating output folder...")
            os.mkdir(self.folder)
            os.mkdir(os.path.join(self.folder, "images"))
            print("Creating images folder...")

        # initialise data parameters
        self.pibot = PenguinPi(args.ip, args.port)

        self.command = {
            "motion": [0, 0],
        }
        self.quit = False
        self.w = DIMS["width"]
        self.h = DIMS["height"]
        self.c = DIMS["channels"]

        if self.mode == "one_shot":
            self.image_id = 0
        else:
            f = open(os.path.join(self.folder, "config.txt"), "r")
            self.image_id = int(f.readline()) + 1
            f.close()
        print(f"Using image ID {self.image_id}...")

        self.img = np.zeros([self.h, self.w, self.c], dtype=np.uint8)

    # wheel control
    def control(self):
        self.pibot.set_velocity(self.command["motion"], tick=20, turning_tick=5)

    # camera control
    def take_pic(self):
        f_ = os.path.join(self.folder, "images", f"img_{self.image_id}.png")
        image = self.pibot.get_image()
        self.img = image[240 - self.h :, :, :]  # crop to 120x320x3
        cv2.imwrite(f_, self.img)
        self.image_id += 1

    def exit(self):
        print("\nExiting program...")
        pygame.quit()
        with open(os.path.join(self.folder, "config.txt"), "w", encoding="utf-8") as f:
            f.write(str(self.image_id))
        self.pibot.stop()
        sys.exit()

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
            self.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar="", type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar="", type=int, default=8080)
    parser.add_argument("--one_shot", metavar="", type=bool, default=False)
    args, _ = parser.parse_known_args()

    canvas = pygame.display.set_mode((700, 700))

    operate = Operate(args)

    RESET = operate.pibot.resetEncoder()

    motor_speeds = []

    """
    Operate.csv format
    | Time steps | Left vel | Right vel | X pos | Y pos | Theta | Total reward | Vel smooth | Pose smooth | Track vis |
    """

    with open(
        os.path.join(operate.folder, "log.csv"), operate.csv_mode, encoding="utf-8"
    ) as f:
        writer = csv.writer(f, delimiter=",")
        while True:
            try:
                operate.update_keyboard()
                operate.control()
                if operate.command["motion"] != [0, 0]:  # actual  input given
                    operate.take_pic()
                    l_vel, r_vel = operate.pibot.getEncoders()
                    x, y, theta = operate.pibot.get_pose()
                    if (
                        operate.mode == "bare"
                    ):  # just log the states, I'll calculate rewards later
                        vals = (operate.image_id, l_vel, r_vel, x, y, theta)
                        writer.writerow(vals)
                    else:  # handle reward calculation and plotting
                        rewards = operate.rewards.reward(
                            l_vel, r_vel, x, y, theta, operate.img
                        )
                        vals = (operate.image_id, l_vel, r_vel, x, y, theta, *rewards)
                        writer.writerow(vals)
                        f.flush()
                        if operate.image_id > 1:
                            operate.plotter.plot_reward()
                            bg = pygame.image.load(operate.reward_plot)
                            canvas.blit(bg, (0, 0))
                RESET = operate.pibot.resetEncoder()
                pygame.display.update()
            except KeyboardInterrupt:
                operate.exit()
                print("Plotting action distribution...\n")
                operate.plotter.plot_action_distribution()
