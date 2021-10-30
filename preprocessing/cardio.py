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

def cm_processing(data):

    ds = [pydicom.dcmread(data[i]) for i in range(len(data))]
    img = [_ds.pixel_array for _ds in ds]
    # to reshape shape of images
    img_shape=config.IMG_SHAPE

    # Crop
    croplist = []
    for i in range(len(img)):
        crop_img = cropBorders(img[i], *config.BREAST['crop'])
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

    data_path=u.get_data_path()
    df = loadImages(data_path)

    nbr_image=len(img)

    for i in range(len(img)):
        save_path = data_path+'\\'+str(i)+'_prepocessed.png'
        cv2.imwrite(filename = save_path, img= padlist[i])

if __name__=='__main__':
    data_path=u.get_data_path()
    df=loadImages(data_path)
    cm_processing(loadImages(data_path))
