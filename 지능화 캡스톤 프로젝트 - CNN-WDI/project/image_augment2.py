import os
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
 



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



image = imageio.imread(a_train_scratch_dir + "\\" + train_scratch_fnames[0])

rotate = iaa.Affine(rotate=(-25, 25))

seq = iaa.Sequential([
    # iaa.Affine(rotate=(-25, 25)),
    # iaa.Fliplr(0.5),
    # iaa.Affine(shear=(-25, 25))
    iaa.Crop(percent=(0, 0.1))
], random_order=True)

images = [image, image, image, image]
# images_aug = rotate(images=images)
images_aug = [seq(image=image) for _ in range(4)]



print("Augmented batch:")
ia.imshow(np.hstack(images_aug))