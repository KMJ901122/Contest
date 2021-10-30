import os,inspect,sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow import keras
import matplotlib.image as im
import pathlib
import cv2
from tensorflow.keras import Model, activations
import math
import config
from preprocessing.common import *
from utils.utils import *

AUTO=tf.data.AUTOTUNE

# Optimizers
lr=config.LR # Hyperparmeter Tuning
rmsprop=tf.keras.optimizers.RMSprop(learning_rate=lr)
adam=tf.keras.optimizers.Adam(learning_rate=lr)

# Callback
checkpoint_path=parent_dir+'/weights/cardio/{epoch:03d}.ckpt'
checkpoint_dir=os.path.dirname(checkpoint_path)
cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, period=10)
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)

def create_model(opt):
    input_tensor=keras.layers.Input((config.IMG_SHAPE, config.IMG_SHAPE, 1))
    inp=keras.layers.Conv2D(3, (3, 3), strides=2, padding='same')(input_tensor)

    with tf.device('/GPU:0'):
        base_model=tf.keras.applications.DenseNet121(include_top=False, weights=None,
                                               classes=2)
    out=base_model(inp)
    out=keras.layers.GlobalAveragePooling2D()(out)
    predictions=keras.layers.Dense(2)(out)

    model=keras.models.Model(inputs=input_tensor, outputs=predictions)
    loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    return model

normal_path=parent_dir+'/datasets/normal'
abnormal_path=parent_dir+'/datasets/abnormal'

normal_image_path=png_data_read(normal_path)
abnormal_image_path=png_data_read(abnormal_path)

print('Number of normal images', len(normal_image_path)) # 1111
print('Number of abnormal images', len(abnormal_image_path)) # 1006
sample_image=im.imread(normal_image_path[0])
print('Shape of a sample image', sample_image.shape)

#-----------------------#

normal_images=[]
abnormal_images=[]

normal_total=len(normal_image_path)
abnormal_total=len(abnormal_image_path)

normal_train=int(normal_total*config.SPLIT)
abnormal_train=int(abnormal_total*config.SPLIT)

# from sklearn.model_selection import ShuffleSplit
# ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
# for train, test in ss.split(normal_image_path):
#     print(normal_image_path[[train]])
#     print(normal_image_path[[test]])

# ---- data setup -----

train_data=train_data_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train)
val_data=val_data_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train)
total_data=total_data_gen(normal_image_path, abnormal_image_path)

train_ds=tf.data.Dataset.from_generator(train_data, (tf.float32, tf.float32), ((config.IMG_SHAPE, config.IMG_SHAPE, 1), (2))).cache().batch(1).prefetch(AUTO)
val_ds=tf.data.Dataset.from_generator(val_data, (tf.float32, tf.float32), ((config.IMG_SHAPE, config.IMG_SHAPE, 1), (2))).cache().batch(1).prefetch(AUTO)

total_ds=tf.data.Dataset.from_generator(total_data, (tf.float32, tf.float32), ((config.IMG_SHAPE, config.IMG_SHAPE, 1), (2))).cache().batch(1).prefetch(AUTO)

datagen=tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=3, height_shift_range=0.03, width_shift_range=0.01, horizontal_flip=True, shear_range=0.05, brightness_range=[0.8, 1.])

alpha=0.2

# ============== train ==============

model=create_model(rmsprop)

if config.TRAIN_WITH_DATAGEN:
    train_with_datagen(model, datagen, train_data, val_ds, batch_size=8, epochs=1, class_weight={0: 1.1, 1: 0.9}, callbacks=[reduce_lr, cp_callback])

if config.TRAIN_WITH_MIXUP:
    train_with_mixup(model, train_data, val_ds, batch_size=8, epochs=1, callbacks=[reduce_lr, cp_callback])

# ----------------------------- confusion matrix test -----------------------------

cf=calc_confusion_matrix(model, val_data)
print(cf)
