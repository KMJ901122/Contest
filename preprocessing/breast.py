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

def bt_processing(data):

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

    # Artefacts removing

    # 1
    own_binarised_img_list = []
    for i in range(len(img)):
        binarised_img = OwnGlobalBinarise(img = normlist[i], **config.BREAST['binary'])
        own_binarised_img_list.append(binarised_img)

    # 2
    edited_mask_list = []
    for i in range(len(img)):
        edited_mask = OpenMask(mask = own_binarised_img_list[i], **config.BREAST['mask'])
        edited_mask_list.append(edited_mask)

    # 3
    X_largest_blobs_list = []
    for i in range(len(img)):
        _, X_largest_blobs = XLargestBlobs(mask = edited_mask_list[i], top_X= 1)
        X_largest_blobs_list.append(X_largest_blobs)

    # 4
    own_masked_img_list = []
    for i in range(len(img)):
        masked_img = ApplyMask(img = normlist[i], mask = X_largest_blobs_list[i])
        own_masked_img_list.append(masked_img)

    # Flip
    fliplist =[]
    for i in range(len(img)):
        horizontal_flip = checkLRFlip(own_masked_img_list[i])
        if horizontal_flip:
            flipped_img = np.fliplr(own_masked_img_list[i])
            fliplist.append(flipped_img)
        else:
            fliplist.append(own_masked_img_list[i])

    # Clahe
    clahelist = []
    for i in range(len(img)):
        clahe_img = clahe(fliplist[i], clip =2.0, tile = (8,8))
        clahelist.append(clahe_img)

    padlist = []
    for i in range(len(img)):
        pad_img = cv2.resize(clahelist[i], dsize =(img_shape, img_shape), interpolation = cv2.INTER_AREA)
        padlist.append(pad_img)

    data_path=u.get_data_path()
    df = loadImages(data_path)

    nbr_image=len(img)

    for i in range(len(img)):
        save_path = data_path+str(i)+'_prepocessed.png'
        cv2.imwrite(filename = save_path, img= padlist[i])

if __name__=='__main__':
    data_path=u.get_data_path()+'\normal'
    df=loadImages(data_path)
    bt_processing(loadImages(data_path))
