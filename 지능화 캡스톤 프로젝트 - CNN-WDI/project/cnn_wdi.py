import os
from os.path import join

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


labelList=['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']

my_dir = os.getcwd()
data_dir = my_dir + "\waferimages"

train_dir = os.path.join(data_dir, 'training')
validation_dir = os.path.join(data_dir, 'testing')

print("------------------------------")
train_none_dir = os.path.join(train_dir, labelList[0])
train_loc_dir = os.path.join(train_dir, labelList[1])
train_edge_loc_dir = os.path.join(train_dir, labelList[2])
train_center_dir = os.path.join(train_dir, labelList[3])
train_edge_ring_dir = os.path.join(train_dir, labelList[4])
train_scratch_dir = os.path.join(train_dir, labelList[5])
train_random_dir = os.path.join(train_dir, labelList[6])
train_near_full_dir = os.path.join(train_dir, labelList[7])
train_donut_dir = os.path.join(train_dir, labelList[8])
print(train_none_dir)
print(train_loc_dir)
print(train_edge_loc_dir)
print(train_center_dir)
print(train_edge_ring_dir)
print(train_scratch_dir)
print(train_random_dir)
print(train_near_full_dir)
print(train_donut_dir)

validation_none_dir = os.path.join(validation_dir, labelList[0])
validation_loc_dir = os.path.join(validation_dir, labelList[1])
validation_edge_loc_dir = os.path.join(validation_dir, labelList[2])
validation_center_dir = os.path.join(validation_dir, labelList[3])
validation_edge_ring_dir = os.path.join(validation_dir, labelList[4])
validation_scratch_dir = os.path.join(validation_dir, labelList[5])
validation_random_dir = os.path.join(validation_dir, labelList[6])
validation_near_full_dir = os.path.join(validation_dir, labelList[7])
validation_donut_dir = os.path.join(validation_dir, labelList[8])
print(validation_none_dir)
print(validation_loc_dir)
print(validation_edge_loc_dir)
print(validation_center_dir)
print(validation_edge_ring_dir)
print(validation_scratch_dir)
print(validation_random_dir)
print(validation_near_full_dir)
print(validation_donut_dir)
print("------------------------------")

train_none_fnames = os.listdir( train_none_dir )
train_loc_fnames = os.listdir( train_loc_dir )
train_edge_loc_fnames = os.listdir( train_edge_loc_dir )
train_center_fnames = os.listdir( train_center_dir )
train_edge_ring_fnames = os.listdir( train_edge_ring_dir )
train_scratch_fnames = os.listdir( train_scratch_dir )
train_random_fnames = os.listdir( train_random_dir )
train_near_full_fnames = os.listdir( train_near_full_dir )
train_donut_fnames = os.listdir( train_donut_dir )

validation_none_fnames = os.listdir( validation_none_dir )
validation_loc_fnames = os.listdir( validation_loc_dir )
validation_edge_loc_fnames = os.listdir( validation_edge_loc_dir )
validation_center_fnames = os.listdir( validation_center_dir )
validation_edge_ring_fnames = os.listdir( validation_edge_ring_dir )
validation_scratch_fnames = os.listdir( validation_scratch_dir )
validation_random_fnames = os.listdir( validation_random_dir )
validation_near_full_fnames = os.listdir( validation_near_full_dir )
validation_donut_fnames = os.listdir( validation_donut_dir )

print('Total training none images :', len(train_none_fnames))
print('Total training loc images :', len(train_loc_fnames))
print('Total training edge_loc images :', len(train_edge_loc_fnames))
print('Total training center images :', len(train_center_fnames))
print('Total training edge_ring images :', len(train_edge_ring_fnames))
print('Total training scratch images :', len(train_scratch_fnames))
print('Total training random images :', len(train_random_fnames))
print('Total training near_full images :', len(train_near_full_fnames))
print('Total training donut images :', len(train_donut_fnames))

print('Total validation none images :', len(validation_none_fnames))
print('Total validation loc images :', len(validation_loc_fnames))
print('Total validation edge_loc images :', len(validation_edge_loc_fnames))
print('Total validation center images :', len(validation_center_fnames))
print('Total validation edge_ring images :', len(validation_edge_ring_fnames))
print('Total validation scratch images :', len(validation_scratch_fnames))
print('Total validation random images :', len(validation_random_fnames))
print('Total validation near_full images :', len(validation_near_full_fnames))
print('Total validation donut images :', len(validation_donut_fnames))

print("------------------------------")

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(24,24,3)),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(3, activation='softmax')
# ])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding="same"),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.SpatialDropout2D(rate=0.2),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(9, activation='softmax')
])

model.summary()

# model.compile(optimizer=RMSprop(lr=0.001),
#             loss='binary_crossentropy',
#             metrics = ['accuracy'])

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics = ['accuracy'])

# train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   horizontal_flip=True, #좌우 대칭
                                   vertical_flip=True, #상하 대칭
                                   width_shift_range=0.3, #좌우 이동, 이동폭/이미지폭
                                   height_shift_range=0.3, #상하 이동
                                   rotation_range=10, #회전 -10 ~ +10, degree (0 ~ 360 도)
                                   shear_range=10, #기울기 -10 ~ 10, degree (0 ~ 360 도)
                                   zoom_range=0.3, #zoom_range=[0.7, 1.3], #확대 축소
                                   featurewise_center=True, #표준 정규화 ([-1, 1])
                                   featurewise_std_normalization=True, #
                                   validation_split=0.3 )

# test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=100,
                                                  class_mode='categorical',
                                                  target_size=(224, 224))
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       batch_size=100,
                                                       class_mode  = 'categorical',
                                                       target_size = (224, 224))

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=20,
                    epochs=20,
                    validation_steps=20,
                    verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()