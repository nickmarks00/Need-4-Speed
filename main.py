# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time
import csv

from tensorflow.keras.models import load_model
import time
import autokeras as ak
from PIL import Image, ImageEnhance
# import matplotlib.pyplot as plt
import torchvision

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot_main import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions


import pygame # python package for GUI
import shutil # python package for file operations

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # robot speed scale factor
        self.speed = 1

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.timer = time.time()
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)

        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.loaded_model = load_model("model_autokeras_two_wheels_6thMarCropped180SaturationSharpness", custom_objects=ak.CUSTOM_OBJECTS) #make sure this model autokeras folder is in the same dir as this ipynb file
        
    # wheel control
    def control(self):       
        predictions = self.loaded_model.predict(self.img)
        predictions = predictions[0]
        # if abs(predictions[0] - predictions[1]) <3:
        #     predictions[1] = predictions[0]
        drive = [0, 0]
        for idx, prediction in enumerate(predictions):
            drive[idx] = int(((40 * prediction) + 7)/1.3) + 10
            
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                drive[0], drive[1])
            print(lv,rv)
            time.sleep(0.5)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)

    # camera control
    def take_pic(self):
        img = self.pibot.get_image()
        print(type(img))
        img = Image.fromarray(img)
        
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(3)
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(4)
        
        img = np.asarray(img).astype('float32') /255.0
        
        img = img[180:240][0:320]
        # img = img[0:320][150:260]
        self.img = np.array([img.reshape(60,320,3)])
        if not self.data is None:
            self.data.write_image(self.img)

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if (self.ekf_on == True): #save images at least after 1 sec
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'
            self.timer = time.time()

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        # ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
        #     not_pause = self.ekf_on)
        # canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        # robot_view = cv2.resize(self.aruco_img, (320, 240))
        # self.draw_pygame_window(canvas, robot_view, 
        #                         position=(h_pad, v_pad)
        #                         )

        # for target detector (M3)
        # detector_view = cv2.resize(self.network_vis,
        #                            (320, 240), cv2.INTER_NEAREST)
        # self.draw_pygame_window(canvas, detector_view, 
        #                         position=(h_pad, 240+2*v_pad)
        #                         )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.ekf_on = True
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()
        
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()

    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                     pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    reset = operate.pibot.resetEncoder()
    
    motor_speeds = []
    
    with open('lab_output/actions.csv', 'w') as f:
        writer = csv.writer(f)
        
        while start:
            operate.update_keyboard()
            operate.take_pic()
            operate.control()
            # operate.save_image()
            # visualise
            # operate.draw(canvas)
            
            if operate.ekf_on:
                left_speed, right_speed = operate.pibot.getEncoders()
                # motor_speeds.append([left_speed, right_speed])
                motor_speeds = [left_speed, right_speed]
                writer.writerow(motor_speeds)
                # print('left speed: ', left_speed, 'right speed: ', right_speed)
            pygame.display.update()
            reset = operate.pibot.resetEncoder()

        



