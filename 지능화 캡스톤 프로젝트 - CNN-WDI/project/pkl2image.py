import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import cv2
import time
import base64
from PIL import Image
import io

# class WM811k():
#     def __init__(self):
#         print("init")
#     def save_image(arr: np.ndarray, filepath: str, vmin: int=0, vmax: int=2):
#         scaled_arr = (arr / vmax) * 255
#         img = Image.fromarray(scaled_arr.astype(np.uint8))
#         img.save(filepath, dpi=(500,500))



#labelList=["Center" ,"Do", "Edge-Loc","Edge-Ring", "Loc", "Ran","Scratch","none","Near"]
labelList=['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']
INDEX_START=0
INDEX_END=100000
FILE_NAME="new_LSWMD_full.tfrecords"
FILE_NAME_NO_LABEL="new_LSWMD_full_NO_label.tfrecords"

print(os.listdir("../CNN-WDI/data/"))

dataset     = pd.read_pickle('../CNN-WDI/data/LSWMD.pkl')
#with open('LSWMD.pkl') as f1:
#   print "loading...,please waitting for few seconds"  

images      = dataset["waferMap"].iloc[:]  
labels      = dataset["failureType"].iloc[:].apply(str)
diesize     = dataset["dieSize"].iloc[:].apply(int)
trianTestLabel = dataset["dieSize"].iloc[:].apply(str)
'''
images      = dataset["waferMap"].iloc[INDEX_START:INDEX_END]  
labels      = dataset["failureType"].iloc[INDEX_START:INDEX_END].apply(str)
diesize     = dataset["dieSize"].iloc[INDEX_START:INDEX_END].apply(int)
trianTestLabel = dataset["dieSize"].iloc[INDEX_START:INDEX_END].apply(str)
'''

#labels=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]

def save_image(arr: np.ndarray, filepath: str, vmin: int=0, vmax: int=255):
    # scaled_arr = (arr / vmax) * 255
    # img = Image.fromarray(scaled_arr.astype(np.uint8))
    # img.save(filepath, dpi=(500,500))

    scaled_arr = (arr / vmax) * 255
    img = Image.fromarray(scaled_arr.astype(np.uint8))
    img.save(filepath)

def get_num(label):
    index_num = 99
    for i in range(0,9):
        if labelList[i] == label:
            index_num = i
    return index_num 

# writer1 = tf.python_io.TFRecordWriter(FILE_NAME)  
# writer2 = tf.python_io.TFRecordWriter(FILE_NAME_NO_LABEL)  

num = len(labels)
print ("the total account of the samples:",num)

names=[]

ilegel_data_count = 0
for i in range(0,num):
    # build tf record
    labels_split = ""
    if (len(labels[i]) < 5):
        labels_split = labels[i].split("'")[0]
  	    # labels[i] = labels[i].split("'")[0]
        
    else:
        labels_split = labels[i].split("'")[1]
  	    # labels[i] = labels[i].split("'")[1]
    
    #if labels[i] not in names:
	    #names.append(labels[i])
    #print "names ",names

    # tmp_num     = get_num(labels[i])
    tmp_num     = get_num(labels_split)
    tmp_imgraw  = images[i].tobytes()  
    tmp_diesize = diesize[i]
    tmp_trianTestLabel = trianTestLabel[i]
    tmp_image_x=images[i].shape[0]
    tmp_image_y=images[i].shape[1]
    #print(image_x,"<__>",image_y)

    if(tmp_num==99):
        
        #print("------------ label ERROR !!! ------------",labels[i],"i: ",i,",size: ",tmp_diesize, "shape:",images[i].shape)
        
        ilegel_data_count = ilegel_data_count + 1 
        example2 = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_imgraw ])),
            'diesize': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_diesize])),
            'image_x': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_image_x])),
            'image_y': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_image_y]))
            
        })) 
        
        # writer2.write(example2.SerializeToString())  # Serialize To String
    else:
        print("---DATA---: ","label: ", labels_split ,"i: ",i,",size: ",tmp_diesize, ", num: ",tmp_num , "shape:",images[i].shape)
        # example1 = tf.train.Example(features=tf.train.Features(feature={
        #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_num])),
        #     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_imgraw])),
        #     'diesize': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_diesize])),
        #     'image_x': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_image_x])),
        #     'image_y': tf.train.Feature(int64_list=tf.train.Int64List(value=[tmp_image_y])),
        #     'trianTestLabel': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_trianTestLabel]))
        # }))

        img = images[i]

        image_path = "data/" + labels_split + "/"
        image_name = labels_split + "_" + str(i)  + ".png"

        save_image(img, image_name)
        
        plt.imshow(img)
        plt.show(block=False)
        # plt.savefig(image_name)
        # cv2.imwrite(image_path+image_name, cv_img)
        plt.pause(0.5)
        plt.close()
        


        
        # writer1.write(example1.SerializeToString())  # Serialize To String

print ("all Done")
print("total data: ",num)
print("data with label: ",num-ilegel_data_count)
print("labeled percentage: ",float((num-ilegel_data_count)/num) )

# writer1.close()
# writer2.close()

'''
print(images)
print('--------')
print(labels)
print('--------')
print(diesize )
print('--------')
img = images[35]
#print(img)
print("-----------------")
plt.imshow( img)#.reshape([60, 41]) )
plt.show()
img = images[45]
plt.imshow( img )
plt.show()
'''