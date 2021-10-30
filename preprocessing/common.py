import os
import numpy as np
import pydicom
import cv2
import pathlib

def loadImages(path):
    image_files = sorted([os.path.join(path, file)
                         for file in os.listdir(path)
                         if file.endswith('.dcm')])
    return image_files

def png_data_read(data_path):
    data_dir=pathlib.Path(data_path)
    png_file_list=data_dir.glob('*.png')
    png_file_list=map(lambda name: str(name), png_file_list)
    return np.array(list(png_file_list))

def jpg_data_read(data_path):
    data_dir=pathlib.Path(data_path)
    png_file_list=data_dir.glob('*.jpg')
    png_file_list=map(lambda name: str(name), png_file_list)
    return np.array(list(png_file_list))

def cropBorders(img, l =0.01, r=0.01, u = 0.04, d =0.04):
    # print('shape of image', img.shape)
    nrows, ncols = img.shape
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1-r))
    u_crop = int(nrows *u)
    d_crop = int(nrows *(1-d))
    cropped_img = img[u_crop: d_crop, l_crop:r_crop]
    return cropped_img

def minMaxNormalise(img):
    norm_img = (img - img.min())/ (img.max() - img.min())
    return norm_img

def OwnGlobalBinarise(img, thresh, maxval):
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img>= thresh] = maxval
    return binarised_img

def OpenMask(mask, ksize =(23,23), operation = "open"):
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = ksize)
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    return edited_mask

def XLargestBlobs(mask, top_X = None):

    def SortContoursByArea(contours, reverse = True):
        sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
        bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
        return sorted_contours, bounding_boxes

    contours, hierarchy = cv2.findContours(image = mask,
                                          mode = cv2.RETR_EXTERNAL,
                                          method = cv2.CHAIN_APPROX_NONE)
    n_contours = len(contours)
    if n_contours > 0 :
        if n_contours < top_X or top_X == None:
            top_X = n_contours

        sorted_contours, bounding_boxes = SortContoursByArea(contours = contours)
        X_largest_contours = sorted_contours[0:top_X]
        to_draw_on = np.zeros(mask.shape, np.uint8)

        X_largest_blobs = cv2.drawContours(image= to_draw_on ,
                                          contours = X_largest_contours,
                                          contourIdx = -1,
                                          color =1,
                                          thickness =-1)
    return n_contours, X_largest_blobs

def ApplyMask(img, mask):
    masked_img = img.copy()
    masked_img[mask==0]=0
    return masked_img

def checkLRFlip(img):
    nrows, ncols = img.shape
    x_center = ncols//2
    y_center = nrows//2

    col_sum = img.sum(axis = 0)
    row_sum = img.sum(axis = 1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center : -1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False
    return LR_flip

def clahe(img, clip =2.0, tile = (8,8)):
    img =cv2.normalize(
        img,
        None,
        alpha =0,
        beta= 255,
        norm_type = cv2.NORM_MINMAX,
        dtype = cv2.CV_32F,
        )
    img_unit8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit = clip, tileGridSize =tile)
    clahe_img = clahe_create.apply(img_unit8)

    return clahe_img

def reversecheck(img):
    number_white = np.sum(img.flatten() > 255//2)
    number_black = np.sum(img.flatten() <=255//2)

    if number_black<number_white:
        reverse = True
    else:
        reverse = False
    return reverse
