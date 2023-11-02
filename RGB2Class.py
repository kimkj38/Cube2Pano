import numpy as np
import os
from PIL import Image

# 주어진 딕셔너리
PALETTE_dict = {(128, 64, 128): 0, (244, 35, 232): 1, (70, 70, 70): 2, (102, 102, 156): 3,
                (190, 153, 153): 4, (153, 153, 153): 5, (250, 170, 30): 6, (220, 220, 0): 7,
                (107, 142, 35): 8, (152, 251, 152): 9, (70, 130, 180): 10, (220, 20, 60): 11,
                (255, 0, 0): 12, (0, 0, 142): 13, (0, 0, 70): 14, (0, 60, 100): 15,
                (0, 80, 100): 16, (0, 0, 230): 17, (119, 11, 32): 18}

def convert_rgb_to_value(rgb_image, palette_dict):
    height, width, _ = rgb_image.shape
    value_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            rgb_tuple = tuple(rgb_image[i, j])
            if rgb_tuple in palette_dict:
                value_image[i, j] = palette_dict[rgb_tuple]

    return value_image

img_src_root = '/root/data2/PanoImgs/train'
label_src_root = '/root/data2/Pano_color_label'

img_dst_root = '/root/data2/cityPano/total_imgs'
label_dst_root = '/root/data2/total_labels_palette'

city_names = os.listdir(img_src_root)

count = 1
for city in city_names:
    city_path = os.path.join(img_src_root, city)
    label_city_path = os.path.join(label_src_root,city)
    
    img_list = os.listdir(city_path)
    for img_name in img_list:
        
        img_path = os.path.join(city_path, img_name)
        label_name = img_path.split('/')[-1].split('_')[:3]
        label_name.append("gtFine_color.png")
        label_name = ("_").join(label_name)
        label_path = os.path.join(label_city_path, label_name)

        # copy(img_path, os.path.join(img_dst_root, '{:05d}.png'.format(count)))
        img = Image.open(label_path)
        img = np.asarray(img)
        
        class_label = Image.fromarray(convert_rgb_to_value(img, PALETTE_dict))
        class_label.save(os.path.join(label_dst_root,'{:05d}.png'.format(count)))
        print(count, os.path.join(label_dst_root,'{:05d}.png'.format(count)))    
        # copy(label_path, os.path.join(label_dst_root,'{:05d}.png'.format(count)))

        count += 1

# files = os.listdir(label_src_root)
# for file in files:
#     label_path = os.path.join(label_src_root, file)
#     img = Image.open(label_path)
#     img = np.asarray(img)
    
#     class_label = Image.fromarray(convert_rgb_to_value(img, PALETTE_dict))
#     class_label.save(os.path.join(label_dst_root,'{:05d}.png'.format(count)))
#     print(count, os.path.join(label_dst_root,'{:05d}.png'.format(count)))    
#     # copy(label_path, os.path.join(label_dst_root,'{:05d}.png'.format(count)))

#     count += 1   