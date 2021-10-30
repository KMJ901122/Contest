import os,inspect,sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pydicom
import cv2
from common import *
import utils.utils as u
import config

def processing(data, save_path):

    img=[cv2.imread(d, cv2.IMREAD_GRAYSCALE) for d in data]
    # to reshape shape of images
    img_shape=config.IMG_SHAPE

    # Crop
    croplist = []
    for i in range(len(img)):
        crop_img = cropBorders(img[i], *config.COMMON['crop'])
        croplist.append(crop_img)

    # Normalise
    normlist = []
    for i in range(len(img)):
        norm_img = minMaxNormalise(croplist[i])
        normlist.append(norm_img)

    # Clahe
    clahelist = []
    for i in range(len(img)):
        clahe_img = clahe(croplist[i], clip =2.0, tile = (8,8))
        clahelist.append(clahe_img)

    # Reverse
    reverselist =[]
    for i in range(len(img)):
        reverse = reversecheck(clahelist[i])
        if reverse:
            reverse_img = np.abs(1 - clahelist[i])
            reverselist.append(reverse_img)
        else:
            reverselist.append(clahelist[i])

    padlist = []
    for i in range(len(img)):
        pad_img = cv2.resize(reverselist[i], dsize =(img_shape, img_shape), interpolation = cv2.INTER_AREA)
        padlist.append(pad_img)


    nbr_image=len(img)

    for i in range(len(img)):
        img_save_path=save_path+'\\'+str(i)+'_preprocessed.png'
        cv2.imwrite(filename = img_save_path, img= padlist[i])

    print('Preprocessed image will be saved as a png file')

if __name__=='__main__':
    base_path=u.get_data_path()
    for add_path in config.DATASET:
        data_path=base_path+'\\'+add_path
        save_path=base_path+'\\'+add_path
        df=jpg_data_read(data_path)
        if len(df)==0:
            print('You have to check that file type is jpg or not')
            print('Now, We address only png file data')
            df=png_data_read(data_path)
        processing(df, save_path)
