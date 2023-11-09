# 1개의 tangent plane 파노라마로

import cv2
import numpy as np
import os
import random
from combineViews import combineViews


############### GT
img_dir = '/root/data1/cityscapes/leftImg8bit/train'
gt_dir = '/root/data1/cityscapes/gtFine/train'
img_save_dir = '/root/data1/SinglePano_imgs'
label_save_dir = '/root/data1/SinglePano_color_labels'

city_list = os.listdir(img_dir)

theta = 5*np.pi/6

# Intrinsic
K = np.zeros((3,3))
K[0,0] = 2262.52 #fx
K[1,1] = 2265.3017905988554 #fy
K[0,2] = 1096.98 #cx
K[1,2] = 513.137 #cy



fx = K[0,0]
cx = K[0,2]
cy = K[1,2]


if not os.path.exists(img_save_dir):
    os.mkdir(img_save_dir)
if not os.path.exists(label_save_dir):
    os.mkdir(label_save_dir)




count = 0
city_list = os.listdir(img_dir)
for city in city_list:
    img_city = os.path.join(img_dir, city)
    gt_city = os.path.join(gt_dir, city)

    img_save_city = os.path.join(img_save_dir, city)
    label_save_city = os.path.join(label_save_dir, city)
    
    if not os.path.exists(img_save_city):
        os.mkdir(img_save_city)
    if not os.path.exists(label_save_city):
        os.mkdir(label_save_city)

    img_list = os.listdir(img_city)
  
    for img_name in img_list:
        # Extrinsic
        roll = 2*np.pi
        # pitch = 1/2*np.pi
        yaw = 5/6*np.pi

        # roll = random.random()*2*np.pi
        # pitch = random.random()*2*np.pi
        rand = random.random()
        rand = (rand+0.5)/1.8
        pitch = rand*np.pi
        # yaw = random.random()*2*np.pi

        R_roll = np.array([[1,0,0],[0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0,1,0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0,0,1]])

        R = R_yaw @ R_pitch @ R_roll

        pose = np.eye(4,4)
        pose[0:3, 0:3] = R

        img_path = os.path.join(img_city, img_name)

        img_name = img_path.split('/')[-1].split('_')[:-1]
        img_name.append("gtFine_color.png")
        gt_name = "_".join(img_name)
        gt_path = os.path.join(gt_city, gt_name)

        img_save_path = os.path.join(img_save_city, img_path.split('/')[-1])
        gt_save_path = os.path.join(label_save_city, gt_name)


        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        h, w, c = img.shape

        w_cx = w-cx

        new_w = round(2 * min(cx, w_cx))

        if w_cx >= cx:
            start_x = 0
        else:
            start_x = w - new_w

        h_cy = h-cy
        new_h = round(2 * min(cy, h_cy))
        if h_cy >= cy:
            start_y = 0
        else:
            start_y = h - new_h

        img = img[start_y: start_y + new_h, start_x: start_x + new_w, :]
        gt = gt[start_y: start_y + new_h, start_x: start_x + new_w, :]

        X, Y = np.meshgrid(np.arange(1, new_w + 1), np.arange(1, new_h + 1))
        X = (X - new_w / 2 - 0.5) / new_w
        Y = (Y - new_h / 2 - 0.5) / new_h
        R = np.sqrt(X ** 2 + Y ** 2 + 1)

        v = np.dot(pose[0:3, 0:3], np.array([[0], [0], [1]]))
        vx = theta + np.arctan2(-v[1], v[0])[0]
        vy = -np.arcsin(v[2])[0]

        sepImg = [{'img': img, 'sz': [new_h, new_w], 'fov': 1.337, 'vx':vx, 'vy':vy}]
        sepGT = [{'img': gt, 'sz': [new_h, new_w], 'fov': 1.337, 'vx':vx, 'vy':vy}]

        panocolor, _ = combineViews(sepImg, 2048*4, 1024*4)
        panoGT, _ = combineViews(sepGT, 2048*4, 1024*4)

        nonzero_coords = np.transpose(np.nonzero(np.any(panocolor != [0, 0, 0], axis=2)))
        left_top = np.min(nonzero_coords, axis=0)
        right_bottom = np.max(nonzero_coords, axis=0)
        panocolor = panocolor[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
        panoGT = panoGT[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]

        cv2.imwrite(img_save_path, panocolor)
        cv2.imwrite(gt_save_path, panoGT)
        count += 1
        print(count, panocolor.shape, img_save_path)