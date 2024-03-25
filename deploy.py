# Package imports
import numpy as np
import sys
import time
import d3rlpy
import torch

# Local imports
from utils.penguin_pi import PenguinPi  # access the robot
from utils.file_handler import fetch_model_path
from utils.globals import DIMS


class Operate:
    def __init__(self, args):
        self.pibot = PenguinPi(args.ip, args.port)
        self.device = "cuda:0" if torch.cuda.is_available() else None

        self.w = DIMS["width"]
        self.h = DIMS["height"]
        self.c = DIMS["channels"]
        self.img = np.zeros(
            [1, self.c, self.w, self.h], dtype=np.uint8
        )  # this is the format that P-Pi images are received as

        path_to_model = fetch_model_path()
        print(f"\nDeploying model {path_to_model}...\n")
        self.model = d3rlpy.load_learnable(path_to_model, device=self.device)

    def take_pic(self):
        img = self.pibot.get_image()
        self.img = img[240 - self.h :, :, :]
        self.img = np.array(self.img, dtype=np.uint8).transpose(2, 1, 0)
        self.img = np.expand_dims(self.img, axis=0)

    def control(self):
        predictions = self.model.predict(self.img)
        print(predictions)
        predictions = predictions[0]
        drive = [0, 0]
        for idx, prediction in enumerate(predictions):
            drive[idx] = int(((40 * prediction) + 7) / 1.3) + 10

        speeds = self.pibot.predict_velocity(drive[0], drive[1])
        print(speeds)
        time.sleep(0.5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar="", type=str, default="192.168.50.1")
    parser.add_argument("--port", metavar="", type=int, default=8080)
    args, _ = parser.parse_known_args()

    start = False
    input = input("\nDo you wish to start deployment? (y/n)... ")
    if input == "y":
        start = True
    else:
        print("Exiting...")
        sys.exit()

    operate = Operate(args)

    reset = operate.pibot.resetEncoder()

    motor_speeds = []

    while start:
        try:
            operate.take_pic()
            operate.control()
            reset = operate.pibot.resetEncoder()
        except KeyboardInterrupt:
            print("Exiting gracefully...")
            print("\nZeroing velocities...")
            operate.pibot.stop()
            sys.exit()
