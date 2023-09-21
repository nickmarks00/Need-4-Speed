# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most three types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = person, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}   
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    #apple_dimensions = [0.075448, 0.074871, 0.071889]
    apple_dimensions = [0.075448, 0.074871, 0.085]
    target_dimensions.append(apple_dimensions)

    #lemon_dimensions = [0.060588, 0.059299, 0.053017]
    lemon_dimensions = [0.060588, 0.059299, 0.065]
    target_dimensions.append(lemon_dimensions)

    #pear_dimensions = [0.0946, 0.0948, 0.135]
    pear_dimensions = [0.0946, 0.0948, 0.145]
    target_dimensions.append(pear_dimensions)

    #orange_dimensions = [0.0721, 0.0771, 0.0739]
    orange_dimensions = [0.0721, 0.0771, 0.089]
    target_dimensions.append(orange_dimensions)

    #strawberry_dimensions = [0.052, 0.0346, 0.0376]
    strawberry_dimensions = [0.052, 0.0346, 0.042]
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    # for target_num in completed_img_dict.keys():
    #     box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]] 
    #     robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
    #     true_height = target_dimensions[target_num-1][2] # true height of the fruit 
        
    #     ######### Replace with your codes #########
    #     # compute pose of the target based on bounding box info and robot's pose
    #     #note box is relative to the robot
        
    #     scale_height = true_height/box[3] # heigh of fruit in pixels
    #     depth = scale_height * focal_length # 
    #     #plane_to_object = depth - focal_length
    #     dispalcement_from_centre_line = depth*(abs(box[0]-camera_matrix[0][2]))/focal_length
        
    #     theta_relative  = np.arctan(dispalcement_from_centre_line/depth) #theta = atan(y/x)
    #     hypotenuse = dispalcement_from_centre_line/np.sin(theta_relative)
    #     theta_miu = 0
    #     if box[0]>=0:
    #         theta_miu = robot_pose[2] - theta_relative
    #     else :
    #         theta_miu = robot_pose[2] + theta_relative
    #     y_pose = hypotenuse*np.sin(theta_miu) + robot_pose[1]
    #     x_pose = hypotenuse*np.cos(theta_miu) + robot_pose[0]
    #     target_pose = {'y':  y_pose  , 'x': x_pose }
        
    #     target_pose_dict[target_list[target_num-1]] = target_pose
    #     ###########################################

    ##Save the largest box
    max_height = 0
    target_num = 0
    for idx in completed_img_dict.keys():
        box = completed_img_dict[idx]['target']
        height = box[3].item()
        if height > max_height:
            max_height = height
            target_num = idx

    box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]] 
    robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
    true_height = target_dimensions[target_num-1][2] # true height of the fruit 
    ######### Replace with your codes #########
    # compute pose of the target based on bounding box info and robot's pose
    #note box is relative to the robot
    
    scale_height = true_height/box[3] # heigh of fruit in pixels
    depth = scale_height * focal_length # 
    #plane_to_object = depth - focal_length
    dispalcement_from_centre_line = depth*(abs(box[0]-camera_matrix[0][2]))/focal_length
    
    # ## Transformations
    # x_prime = depth
    # y_prime = -dispalcement_from_centre_line
    # coords_prime = np.array([x_prime, y_prime, 0, 1]).T

    # theta_r = robot_pose[2]

    # rot_z = np.identity(4)
    # rot_z[0:2, 0:2] =   [[np.cos(theta_r), -np.sin(theta_r)],
    #                     [np.sin(theta_r), np.cos(theta_r)]]

    # trans = np.identity(4)
    # trans[0, -1] = robot_pose[0]
    # trans[1,-1] = robot_pose[1]

    # transformation = trans @ rot_z
    # print('tranformation', transformation)
    
    # coords = transformation @ coords_prime
    # print('new', coords)
    # # x_pose = coords[0]
    # # y_pose = coords[1]

    
    theta_relative  = np.arctan(dispalcement_from_centre_line/depth) #theta = atan(y/x)
    hypotenuse = dispalcement_from_centre_line/np.sin(theta_relative)
    theta_miu = 0
    if box[0]>=0:
        theta_miu = robot_pose[2] - theta_relative
    else :
        theta_miu = robot_pose[2] + theta_relative
    y_pose = hypotenuse*np.sin(theta_miu) + robot_pose[1]
    x_pose = hypotenuse*np.cos(theta_miu) + robot_pose[0]

    target_pose = {'y':  y_pose  , 'x': x_pose }
    target_pose_dict[target_list[target_num-1]] = target_pose

    

        

    return target_pose_dict

def get_n_clusters():
    search_list = {
        "apple" : 0,
        "lemon" : 0,
        "orange" : 0,
        "pear" : 0,
        "strawberry" : 0
    }
    with open('../M5_inputs/sim/search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list[fruit.strip()] += 1
        for key in search_list.keys():
            if search_list[key] == 0:
                search_list[key] += 2
    return search_list

def k_means_cluster(fruit_list, n_clusters = 2):
    X = np.array(fruit_list).reshape(len(fruit_list),2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return list(kmeans.cluster_centers_)

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}
    
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    search_list = get_n_clusters()
    if len(apple_est) > search_list["apple"]:
        apple_est = k_means_cluster(apple_est,search_list["apple"])
    elif not apple_est:
        apple_est = [[0., 0.], [0., 0.]]
    else:
        apple_est.append([0., 0.])

    if len(lemon_est) > search_list["lemon"]:
        lemon_est = k_means_cluster(lemon_est,search_list["lemon"])
    elif not lemon_est:
        lemon_est = [[0., 0.], [0., 0.]]
    else:
        lemon_est.append([0., 0.])

    if len(pear_est) > search_list["pear"]:
        pear_est = k_means_cluster(pear_est,search_list["pear"])
    elif not pear_est:
        pear_est = [[0., 0.], [0., 0.]]
    else:
        pear_est.append([0., 0.])

    if len(orange_est) > search_list["orange"]:
        orange_est = k_means_cluster(orange_est, search_list["orange"])
    elif not orange_est:
        orange_est = [[0., 0.], [0., 0.]]
    else:
        orange_est.append([0., 0.])

    if len(strawberry_est) > search_list["strawberry"]:
        strawberry_est = k_means_cluster(strawberry_est, search_list["strawberry"])
    elif not strawberry_est:
        strawberry_est = [[0., 0.], [0., 0.]]
    else:
        strawberry_est.append([0., 0.])

    for i in range(2):
        try:
            target_est['apple_'+str(i)] = {'y':np.asscalar(apple_est[i][0]), 'x':np.asscalar(apple_est[i][1])}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':np.asscalar(lemon_est[i][0]), 'x':np.asscalar(lemon_est[i][1])}
        except:
            pass
        try:
            target_est['pear_'+str(i)] = {'y':np.asscalar(pear_est[i][0]), 'x':np.asscalar(pear_est[i][1])}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'y':np.asscalar(orange_est[i][0]), 'x':np.asscalar(orange_est[i][1])}
        except:
            pass
        try:
            target_est['strawberry_'+str(i)] = {'y':np.asscalar(strawberry_est[i][0]), 'x':np.asscalar(strawberry_est[i][1])}
        except:
            pass
    ###########################################
        
    return target_est

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", metavar='', type=str, default='sim')
    args, _ = parser.parse_known_args()
    # camera_matrix = np.ones((3,3))/2
    fileK = "../M5_inputs/{}/param/intrinsic.txt".format(args.mode)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    print(camera_matrix)
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    print('target_est',target_est)
    # save target pose estimations
    with open(base_dir / 'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, default=default)
    
    print('Estimations saved!')



