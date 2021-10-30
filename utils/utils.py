import os,inspect,sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pandas as pd
import pathlib
import matplotlib.image as im
import numpy as np
import tensorflow as tf
import config
from sklearn.metrics import confusion_matrix
from collections import defaultdict

def get_data_path():
    path_list=sys.path[0].split('\\')[:-1]
    data_path=path_list[0]
    for p in path_list[1:]:
        data_path+='\\'+p
    data_path=data_path+'\datasets'
    return data_path

def png_data_read(data_path):
    data_dir=pathlib.Path(data_path)
    png_file_list=data_dir.glob('*.png')
    png_file_list=map(lambda name: str(name), png_file_list)
    return list(png_file_list)

def mix_up(ds_one, ds_two, alpha=0.2):
    images_one, labels_one=ds_one
    images_two, labels_two=ds_two
    batch_size=tf.shape(images_one)[0]

    def sample_beta_distribution(size, c0=alpha, c1=alpha):
        g1_sample=tf.random.gamma(shape=[size], alpha=c1)
        g2_sample=tf.random.gamma(shape=[size], alpha=c0)
        return g1_sample/(g1_sample+g2_sample)

    l=sample_beta_distribution(batch_size, alpha, alpha)
    x_l=tf.reshape(l, (batch_size, 1, 1, 1))
    y_l=tf.reshape(l, (batch_size, 1))

    images=images_one*x_l+images_two*(1-x_l)
    labels=labels_one*y_l+labels_two*(1-y_l)
    return (images, labels)

def train_data_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train):
    def gen():
        for i in normal_image_path[:normal_train]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([0., 1.]).astype(np.float32) # normal data
            yield (out, label)
        for i in abnormal_image_path[:abnormal_train]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32)
            yield (out, label)
    return gen

def multi_train_data_gen(normal_image_path, pneu_image_path, covid_image_path, x_image_path, y_image_path, split_nbr=config.SPLIT):
    def gen():
        for i in normal_image_path[:int(len(normal_image_path)*split_nbr)]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([0., 1.]).astype(np.float32) # normal data
            yield (out, label)
        for i in abnormal_image_path[:int(len(pneu_image_path)*split_nbr)]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32)
            yield (out, label)
        for i in covid_image_path[:int(len(covid_image_path)*split_nbr)]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([0., 1.]).astype(np.float32) # normal data
            yield (out, label)
        for i in x_image_path[:int(len(x_image_path)*split_nbr)]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32)
            yield (out, label)
        for i in y_image_path[:int(len(y_image_path)*split_nbr)]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32)
            yield (out, label)
    return gen

def val_data_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train):
    def gen():
        for i in normal_image_path[normal_train:]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([0., 1.]).astype(np.float32) # normal data
            yield (out, label)
        for i in abnormal_image_path[abnormal_train:]:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32) # abnormal data
            yield (out, label)
    return gen

def total_data_gen(normal_image_path, abnormal_image_path):
    def gen():
        for i in normal_image_path:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([.0, 1.]).astype(np.float32)
            yield (out, label)
        for i in abnormal_image_path:
            out=im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1)
            out=out.astype(np.float32)
            label=np.array([1., 0.]).astype(np.float32)
            yield (out, label)
    return gen

def mg_train_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train):
    def gen():
        for i in range(normal_train):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in normal_image_path[4*i: 4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([0, 1]).astype(np.float32)
            yield (out, label)
        for i in range(abnormal_train):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in abnormal_image_path[4*i: 4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([1, 0]).astype(np.float32)
            yield (out, label)
    return gen

def mg_val_gen(normal_image_path, abnormal_image_path, normal_train, abnormal_train):
    def gen():
        val_normal_train=(len(normal_image_path)-4*normal_train)//4
        for i in range(val_normal_train):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in normal_image_path[4*normal_train+4*i: 4*normal_train+4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([0, 1]).astype(np.float32)
            yield (out, label)
        val_abnormal_train=(len(abnormal_image_path)-4*abnormal_train)//4
        for i in range(val_abnormal_train):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in abnormal_image_path[4*abnormal_train+4*i: 4*abnormal_train+4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([1, 0]).astype(np.float32)
            yield (out, label)
    return gen

def mg_total_gen(normal_image_path, abnormal_image_path):
    def gen():
        for i in range(len(normal_image_path)):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in normal_image_path[4*i: 4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([0, 1]).astype(np.float32)
            yield (out, label)
        for i in range(len(abnormal_image_path)):
            temp=[im.imread(i).reshape(config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in abnormal_image_path[4*i: 4*i+4]]
            out=np.concatenate(temp, axis=-1)
            out=out.astype(np.float32)
            label=np.array([1, 0]).astype(np.float32)
            yield (out, label)
    return gen

def train_with_mixup(model, train_data, val_data, batch_size=8, epochs=1, weight=None, callbacks=None):
    AUTO=tf.data.AUTOTUNE
    train_ds_one=tf.data.Dataset.from_generator(train_data, (tf.float32, tf.float32), ((config.IMG_SHAPE, config.IMG_SHAPE, 1), (2))).cache().shuffle(100).batch(1).prefetch(AUTO)
    train_ds_two=tf.data.Dataset.from_generator(train_data, (tf.float32, tf.float32), ((config.IMG_SHAPE, config.IMG_SHAPE, 1), (2))).cache().shuffle(100).batch(1).prefetch(AUTO)
    train_ds=tf.data.Dataset.zip((train_ds_one, train_ds_two))
    alpha=0.2
    train_ds_mu=train_ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha), num_parallel_calls=AUTO)

    model.fit(train_ds_mu, validation_data=val_data, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)

def train_with_datagen(model, datagen, train_data, val_data, batch_size=16, epochs=1, class_weight=None, weight=None, callbacks=None):
    x_train=[]
    x_label=[]

    for i, j in train_data():
        x_label.append(j)
        x_train.append(i)

    x_train=np.array(x_train)
    x_label=np.array(x_label)

    model.fit(datagen.flow(x_train, x_label, batch_size=batch_size), class_weight=class_weight, validation_data=val_data, verbose=2, epochs=epochs, callbacks=callbacks)

def calc_confusion_matrix(model, val_data):
    x_test=[]
    test_label=[]

    for img, label in val_data():
        x_test.append(img)
        if list(label)==[0., 1.]:
            label=0 # normal
        elif list(label)==[1., 0.]:
            label=1 # abnormal
        test_label.append(label)

    x_test=np.array(x_test)
    test_label=np.array(test_label)

    pred=model.predict(x_test)
    pred=np.argmax(pred, axis=1)
    cf=confusion_matrix(pred, test_label)

    return cf

def create_csv_file(model, img_path, weight_nbr, weight_path, save_path):
    data_to_csv=defaultdict(list)
    images=[im.imread(i).reshape(1, config.IMG_SHAPE, config.IMG_SHAPE, 1) for i in img_path]
    predictions=[]
    for i in range(len(images)):
        for j in weight_nbr:
            img=images[i]
            w=weight_path+str(j)+'.h5'
            model.load_weights(w)
            prediction=model.predict(img)
            prediction=tf.nn.softmax(prediction)
            predictions.append(prediction)
            mean_pred=tf.math.reduce_mean(predictions, axis=0)
            result=np.argmax(mean_pred)
        img_name=img_path[i].split('\\')[-1]
        data_to_csv['ID'].append(img_name) # should be designated!
        data_to_csv['result'].append(result)

    final_csv=pd.DataFrame(data_to_csv)
    final_csv.set_index('ID', inplace=True)
    final_csv.to_csv(save_path)
