# 파노라마 이미지에서 tangent plane으로
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2

from torchvision import transforms
import math
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype=np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz


def equi2pers(erp_img, fov, nrows, patch_size, label=False):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w / 360.0, fov_h / 180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

    if nrows == 4:
        num_rows = 4
        num_cols = [3, 6, 6, 3]
        phi_centers = [-67.5, -22.5, 22.5, 67.5]
    if nrows == 6:
        num_rows = 6
        num_cols = [3, 8, 12, 12, 8, 3]
        phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
    if nrows == 3:
        num_rows = 3
        num_cols = [3, 4, 3]
        # phi_centers = [-60, 0, 60]
        phi_centers = [-40, 0, 40]
    if nrows == 5:
        num_rows = 5
        num_cols = [3, 6, 8, 6, 3]
        phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
    if nrows == 1:
        num_rows = 1
        num_cols = [6]
        phi_centers = [0]

    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2

            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos)
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  # 0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  # 0 to 1

    cp = center_point * 2 - 1
    center_p = cp.clone()
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c,
                                    rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2
    lon_new = lon / PI
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new < -1] += 2

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch * width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch * width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    if label:
        pers = F.grid_sample(erp_img, grid, mode='nearest', padding_mode='border', align_corners=True)
    else:
        pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)

    grid_tmp = torch.stack([lon, lat], -1)
    xyz = uv2xyz(grid_tmp)
    xyz = xyz.reshape(num_patch, height, width, 3).transpose(0, 3, 1, 2)
    xyz = torch.from_numpy(xyz).to(pers.device).contiguous()

    uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
    uv = uv.contiguous()
    
    return pers, xyz, uv, center_p


# WildPASS
# src_root = '/root/data2/WildPASS/total_imgs'
# dst_root = '/root/data2/WildPASS/Tangents'

# count = 1
# img_names = os.listdir(src_root)
# for n, name in enumerate(img_names):
#     print(n)
#     erp_pad = np.zeros((1024, 2048, 3))
#     gt_pad = np.zeros((1024,2048))
    
#     erp_path = os.path.join(src_root, name)
#     erp = Image.open(erp_path)
    
#     erp_pad[312:712,:] = erp
#     transform = transforms.ToTensor()
#     erp_pad = transform(erp_pad)
    
#     pers, xyz, uv, center_p = equi2pers(erp_pad.unsqueeze(0).float(), 70, 1, (512,512))
    
#     for i in range(6):
#         per = pers[...,i].squeeze(0).permute(1,2,0)
#         per = per.numpy().astype(np.uint8)
#         per = Image.fromarray(per)
#         per.save(os.path.join(dst_root, '{}.jpg'.format(count)))
#         count += 1



# # DensePASS
# src_root = '/root/data2/DensePASS/leftImg8bit/val'
# src_label_root = '/root/data2/DensePASS/gtFine/val'
# dst_root = '/root/data2/DensePASS/Tangents'
# dst_label_root = '/root/data2/DensePASS/Tangents_label'

# count = 1
# img_names = os.listdir(src_root)
# for n, name in enumerate(img_names):
#     print(n)
#     erp_pad = np.zeros((1024, 2048, 3))
#     gt_pad = np.zeros((1024,2048))
    
#     erp_path = os.path.join(src_root, name)
#     erp = Image.open(erp_path)
    
#     gt_path = os.path.join(src_label_root, name.replace('_.png', '_labelTrainIds.png'))
#     gt = Image.open(gt_path)
 
#     erp_pad[312:712,:] = erp
#     gt_pad[312:712,:] = gt
#     transform = transforms.ToTensor()
#     erp_pad = transform(erp_pad)
#     gt_pad = transform(gt_pad)
    
#     pers, xyz, uv, center_p = equi2pers(erp_pad.unsqueeze(0).float(), 70, 1, (512,512))
#     pers_gt,xyz_gt, uv_gt, center_p_gt = equi2pers(gt_pad.unsqueeze(0).float(), 70, 1, (512,512), label=True)
    
    
#     for i in range(6):
#         per = pers[...,i].squeeze(0).permute(1,2,0)
#         per = per.numpy().astype(np.uint8)
#         per = Image.fromarray(per)
#         per.save(os.path.join(dst_root, '{}.png'.format(count)))
        
#         per_gt = pers_gt[...,i].squeeze(0).permute(1,2,0)
#         per_gt = per_gt.numpy().astype(np.uint8)
#         cv2.imwrite(os.path.join(dst_label_root, '{}.png'.format(count)), per_gt)
        
#         count += 1


# SF2D3D
src_root = '/root/data2/SF2D3D/imgs/test'
src_label_root = '/root/data2/SF2D3D/labels7/test'
dst_root = '/root/data2/SF_Tangents/imgs/test'
dst_label_root = '/root/data2/SF_Tangents/labels7/test'

count = 1
img_names = os.listdir(src_root)
for n, name in enumerate(img_names):
    print(n)

    erp_path = os.path.join(src_root, name)
    erp = Image.open(erp_path)
    erp = np.asarray(erp)
    
    gt_path = os.path.join(src_label_root, name.replace('.png', '_labelTrainIds.png'))
    gt = Image.open(gt_path)
    gt = np.asarray(gt)

    transform = transforms.ToTensor()
    erp = transform(erp)
    gt = transform(gt)
    
    pers, xyz, uv, center_p = equi2pers(erp.unsqueeze(0).float(), 80, 3, (512,512))
    pers_gt,xyz_gt, uv_gt, center_p_gt = equi2pers(gt.unsqueeze(0).float(), 80, 3, (512,512), label=True)
    
    
    for i in range(10):
        per = pers[...,i].squeeze(0).permute(1,2,0)
        per = (per.numpy()*255).astype(np.uint8)
        per = Image.fromarray(per)
        per.save(os.path.join(dst_root, '{}.png'.format(count)))
        
        per_gt = pers_gt[...,i].squeeze(0).permute(1,2,0)
        per_gt = (per_gt.numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_label_root, '{}.png'.format(count)), per_gt)
        
        count += 1
    
    
    
    
    
    

# erp_pad = np.zeros((1024, 2048, 3))
# gt_pad = np.zeros((1024,2048))

# erp_path = '/root/data2/DensePASS/leftImg8bit/val/46_.png'
# gt_path = '/root/data2/DensePASS/gtFine/val/46_labelTrainIds.png'

# erp = Image.open(erp_path)
# erp_pad[312:712,:] = erp

# gt = Image.open(gt_path)
# gt_pad[312:712,:] = gt

# transform = transforms.ToTensor()
# erp_pad = transform(erp_pad)
# gt_pad = transform(gt_pad)