import os
from os.path import join
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

import Augmentor

def save_image(arr: np.ndarray, filepath: str, vmin: int=0, vmax: int=255):
    # scaled_arr = (arr / vmax) * 255
    # img = Image.fromarray(scaled_arr.astype(np.uint8))
    # img.save(filepath, dpi=(500,500))

    scaled_arr = (arr / vmax) * 255
    img = Image.fromarray(scaled_arr.astype(np.uint8))
    img.save(filepath)


labelList=['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']

my_dir = os.getcwd()
data_augment_dir = my_dir + "\waferimages_augment"

a_train_dir = os.path.join(data_augment_dir, 'training')

a_train_none_dir = os.path.join(a_train_dir, labelList[0])
a_train_loc_dir = os.path.join(a_train_dir, labelList[1])
a_train_edge_loc_dir = os.path.join(a_train_dir, labelList[2])
a_train_center_dir = os.path.join(a_train_dir, labelList[3])
a_train_edge_ring_dir = os.path.join(a_train_dir, labelList[4])
a_train_scratch_dir = os.path.join(a_train_dir, labelList[5])
a_train_random_dir = os.path.join(a_train_dir, labelList[6])
a_train_near_full_dir = os.path.join(a_train_dir, labelList[7])
a_train_donut_dir = os.path.join(a_train_dir, labelList[8])


train_none_fnames = os.listdir( a_train_none_dir )
train_loc_fnames = os.listdir( a_train_loc_dir )
train_edge_loc_fnames = os.listdir( a_train_edge_loc_dir )
train_center_fnames = os.listdir( a_train_center_dir )
train_edge_ring_fnames = os.listdir( a_train_edge_ring_dir )
train_scratch_fnames = os.listdir( a_train_scratch_dir )
train_random_fnames = os.listdir( a_train_random_dir )
train_near_full_fnames = os.listdir( a_train_near_full_dir )
train_donut_fnames = os.listdir( a_train_donut_dir )


# if(len(train_none_fnames) > 2000):
#     list_a = range(0, len(train_none_fnames))
   
#     random.shuffle(list_a)
    
#     for i in range(0,2000):
#         loc = a_train_none_dir +'\\' + str(i) + '.png'
#         img = Image.open(train_none_fnames[list_a[i]])
#         plt.imshow(img)
# 		# plt.savefig(loc,bbox_inches='tight')
# 		# plt.clf()





## 증강 시킬 이미지 폴더 경로
img = Augmentor.Pipeline(a_train_none_dir)

img.skew()
## 좌우 반전
img.flip_left_right(probability=1.0) 

## 상하 반전
img.flip_top_bottom(probability=1.0)

## 왜곡
img.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8)

## 증강 이미지 수
img.sample(10)