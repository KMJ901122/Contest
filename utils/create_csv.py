import os,inspect,sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
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
from collections import defaultdict
from preprocessing.common import *
from utils import *
data_name=sys.argv[1]
print(data_name)
if data_name not in config.DATASET:
    print(data_name)
    raise ValueError("you need to check the name of data")

AUTO=tf.data.AUTOTUNE

# Optimizers
lr=config.LR # Hyperparmeter Tuning
rmsprop=tf.keras.optimizers.RMSprop(learning_rate=lr)
adam=tf.keras.optimizers.Adam(learning_rate=lr)

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

model=create_model(rmsprop)

weight_nbr=[0, ] # you can ensembel with diferent weight
weight_path=parent_dir+'/weights/'+data_name+'//'
save_path=parent_dir+'/result_csv_files/'+data_name+'.csv'
test_image_path=parent_dir+'/datasets/'+data_name+'_test'
test_image=png_data_read(test_image_path)
create_csv_file(model, test_image, weight_nbr, weight_path, save_path)
