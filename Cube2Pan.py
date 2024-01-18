import cv2
import numpy as np
import os
from combineViews import combineViews


######## Image
# root_dir = '/root/data1/cityscapes/leftImg8bit/train'
# save_dir = '/root/data1/PanoImgs/train'

# city_list = os.listdir(root_dir)



# vx = [-np.pi / 2, -np.pi / 2, 0, np.pi / 2, np.pi, -np.pi / 2]
# vy = [np.pi / 2, 0, 0, 0, 0, -np.pi / 2]

# count = 0
# for city in city_list:
#     save_city = os.path.join(save_dir, city)
#     city_dir = os.path.join(root_dir, city)
    
#     if not os.path.exists(save_city):
#         os.mkdir(save_city)

#     img_list = os.listdir(city_dir)
    
#     for img_name in img_list:
#         img_path = os.path.join(city_dir, img_name)
#         img = cv2.imread(img_path)[:1024,:1024]
#         sepImg = []
#         for i in range(6):
            

#             sepImg.append({
#                 'img': cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX),
#                 # 'img': img,\
#                 'vx': vx[i],
#                 'vy': vy[i],
#                 'fov': np.pi / 2 + 0.001,
#                 'sz': img.shape
#             })

#         panoskybox, _ = combineViews(sepImg, 2048, 1024)
#         # panoskybox = (panoskybox / np.max(panoskybox) * 255).astype(np.uint8)
#         save_name = img_path.split('/')[-1].split('.')[0] + "_Pano.png"
#         print(count, f"{save_city}/{save_name}")
#         cv2.imwrite(f"{save_city}/{save_name}", panoskybox*255)
#         count += 1


######## Depth (combineViews.py에서 interpolation 방식 nearest로)

# ref_dir = '/root/data1/cityscapes/leftImg8bit/train'
# root_dir = '/root/data1/cityscapes/gtFine/train'
# save_dir = '/root/data1/Pano_color_label'

# city_list = os.listdir(root_dir)



# vx = [-np.pi / 2, -np.pi / 2, 0, np.pi / 2, np.pi, -np.pi / 2]
# vy = [np.pi / 2, 0, 0, 0, 0, -np.pi / 2]

# count = 0
# for city in city_list:
#     save_city = os.path.join(save_dir, city)
#     city_dir = os.path.join(ref_dir, city)
    
#     if not os.path.exists(save_city):
#         os.mkdir(save_city)

#     img_list = os.listdir(city_dir)
#     gt_root = os.path.join(root_dir, city)
    
#     for img_name in img_list:
        
#         img_path = os.path.join(city_dir, img_name)
#         img_name = img_path.split('/')[-1].split('_')[:-1]
#         img_name.append("gtFine_color.png")
#         gt_name = "_".join(img_name)
#         gt_path = os.path.join(gt_root, gt_name)
#         save_path = f"{save_city}/{gt_name}"
        
#         if not os.path.exists(save_path):
#             gt = cv2.imread(gt_path)[:1024,:1024]

            
            
#             sepImg = []
#             for i in range(6):
                

#                 sepImg.append({
#                     'img': gt,
#                     'vx': vx[i],
#                     'vy': vy[i],
#                     'fov': np.pi / 2 + 0.001,
#                     'sz': gt.shape
#                 })

#             panoskybox, _ = combineViews(sepImg, 2048, 1024)

#             cv2.imwrite(save_path, panoskybox)
#             print(count, save_path)
#         count += 1



# # cityscape Cubemap 이미지들 파노라마로
# img_root = './cube_sample'
# save_path = './cube_sample.png'

# # vx = [-np.pi / 2, -np.pi / 2, 0, np.pi / 2, np.pi, -np.pi / 2]
# vx = [-np.pi, -np.pi, -np.pi/2, 0, np.pi/2, -np.pi]
# vy = [np.pi / 2, 0, 0, 0, 0, -np.pi / 2]


            

# sepImg = []
# for i in range(6):
#     img = cv2.imread(os.path.join(img_root, '{}.png'.format(i)))

#     sepImg.append({
#         'img': cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX),
#         'vx': vx[i],
#         'vy': vy[i],
#         'fov': np.pi / 2 + 0.001,
#         'sz': img.shape
#     })

# panoskybox, _ = combineViews(sepImg, 2048, 1024)

# cv2.imwrite(save_path, panoskybox*255)


########## Single CS 4개의 tangent 이미지로 쪼개서 cubemap으로

img_path = '/Users/kyeongjun/Desktop/Cube2Pano/city_data/bremen_000040_000019_leftImg8bit.png'
save_path = '/Users/kyeongjun/Desktop/Cube2Pano/output/CS_Tan4/tan4.png'

# vx = [-np.pi / 2, -np.pi / 2, 0, np.pi / 2, np.pi, -np.pi / 2]
# vx = [-np.pi, -np.pi, -np.pi/2, 0, np.pi/2, -np.pi]
# vy = [np.pi / 2, 0, 0, 0, 0, -np.pi / 2]

vx = [-np.pi*3/4, -np.pi/4, np.pi/4, np.pi*3/4]
vy = [0, 0, 0, 0]
            
img = cv2.imread(img_path)
img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
sepImg = []
for i in range(4):

    sepImg.append({
        'img': img[256:768, i*512:(i+1)*512],
        'vx': vx[i],
        'vy': vy[i],
        'fov': np.pi / 2 + 0.001,
        'sz': img.shape
    })

panoskybox, _ = combineViews(sepImg, 2048, 1024)

cv2.imwrite(save_path, panoskybox*255)
