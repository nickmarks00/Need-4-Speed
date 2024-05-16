import numpy as np
import requests
import cv2
import sys


class PenguinPi:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]
        self.endpoint = "http://{}:{}".format(self.ip, self.port)

    ##########################################
    # Change the robot velocity here
    # tick = forward speed
    # turning_tick = turning speed
    ##########################################
    def set_velocity(self, command, tick=20, turning_tick=5, time=0):
        l_vel = command[0] * tick - command[1] * turning_tick
        r_vel = command[0] * tick + command[1] * turning_tick
        self.wheel_vel = [l_vel, r_vel]
        if time == 0:
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value="
                + str(l_vel)
                + ","
                + str(r_vel)
            )
        else:
            assert time > 0, "Time must be positive."
            assert time < 30, "Time must be less than network timeout (20s)."
            requests.get(
                "http://"
                + self.ip
                + ":"
                + str(self.port)
                + "/robot/set/velocity?value="
                + str(l_vel)
                + ","
                + str(r_vel)
                + "&time="
                + str(time)
            )

    def predict_velocity(self, l_vel, r_vel, time=0.25):
        self.wheel_vel = [l_vel, l_vel]
        if time == 0:
            requests.get(
                f"http://{self.ip}:{self.port}/robot/set/velocity?value="
                + str(l_vel)
                + ","
                + str(r_vel)
            )
        else:
            assert time > 0, "Time must be positive."
            assert time < 30, "Time must be less than network timeout (20s)."
            requests.get(
                "http://"
                + self.ip
                + ":"
                + str(self.port)
                + "/robot/set/velocity?value="
                + str(l_vel)
                + ","
                + str(r_vel)
                + "&time="
                + str(time)
            )
        return l_vel, r_vel

    def get_image(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/camera/get")
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            print(f"Image retrieval timed out, {e}")
            img = np.zeros((240, 320, 3), dtype=np.uint8)
        return img

    def getEncoders(self):
        resp = requests.get("{}/robot/get/encoder".format(self.endpoint), timeout=1)
        left_enc, right_enc = resp.text.split(",")
        if abs(int(left_enc)) > 100 or abs(int(right_enc)) > 100:
            raise ValueError("Received extremely large encoder value, ignoring...")
        return int(left_enc), int(right_enc)

    def get_pose(self):
        resp = requests.get("{}/robot/get/pose".format(self.endpoint), timeout=1)
        assert resp.status_code == 200
        x, y, theta = list(
            map(float, resp.text.split(","))
        )  # read str,str,str into float,float,float
        return x, y, theta

    def resetEncoder(self):
        try:
            _ = requests.get("{}/robot/hw/reset".format(self.endpoint), timeout=5)
            return True
        except requests.exceptions.Timeout as _:
            print(
                "Timed out attempting to communicate with {}:{}".format(
                    self.ip, self.port
                ),
                file=sys.stderr,
            )
            return False

    def stop(self):
        try:
            resp = requests.get("{}/robot/stop".format(self.endpoint), timeout=1)
            return resp.json()
        except requests.exceptions.Timeout as _:
            print(
                "Timed out attempting to communicate with {}:{}".format(
                    self.ip, self.port
                )
            )
            return None
