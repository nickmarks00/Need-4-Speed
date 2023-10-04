from typing import Dict, Optional
from typing import Any
import numpy as np
import gymnasium as gym

from gymnasium.spaces import Box

import os
import random
import cv2
import math


# Get the list of all files and directories
# in the root directory

class ObservationMatchingEnv(gym.Env):
    def __init__(self, num_options: int = 2):
        self.state = None
        self.path = "/home/david/Desktop/Dataset/"
        self.pic_id = 0
        self.dir_list = os.listdir(self.path)
        self.num_options = num_options
        self.observation_space = Box(0, 255, shape=(3,320,240)) #RGB image
        self.action_space = Box(-100, 100, shape= (1,2)) # box (low high shape)?

    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None):
        #super().reset(seed=seed, options=options)
        self.pic_id = random.randint(0,len(self.dir_list)-1)

        pathToPic = "img_" + str(self.pic_id) + ".png"
        self.state = cv2.imread(pathToPic)
        return self.state
    def interpolate(picture, action):
        pass

    def findClosest(self,action):
        #given action, interpolate picture

        interpolatedPic =  self.interpolate(self.state, action)#find the interpolated picture

        closestID = None
        currentMinDistance = math.inf
        #might be good to crop the image to the lower half of the picture
        for i in range(0,len(self.dir_list)-1):
            currentPicIteration = cv2.imread("img_" + str(i) + ".png")
            distance = ((interpolatedPic - currentPicIteration).mean()).mean()
            if distance < currentMinDistance:
                currentMinDistance = distance
                closestID = i
        
        self.pic_id = closestID
        

    def step(self, action):
        
        #next immediate encoder 
        nextEncoderValue = ___

        #find next state given an action, return the CLOSEST picture of the interpolated image
        self.findClosest(action)
        pathToPic = "img_" + str(self.pic_id) + ".png"
        self.state = cv2.imread(pathToPic)

        #define a reward, get the difference
        reward = [0,0]
        for i in range(0,2):
            if action[i] > nextEncoderValue[i]:
                reward[i] = abs(action[i] - nextEncoderValue[i])
            else:
                reward[i] = abs(nextEncoderValue[i] - action[i])

        reward = -1*reward.mean()
        
        return self.state, reward, False, False, {}
            
     