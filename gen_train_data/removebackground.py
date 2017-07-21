import os
import argparse
import numpy as np
import scipy
import cv2

from skimage import data, io
from skimage import exposure, img_as_uint, img_as_float, transform
from scipy import ndimage
from PIL import Image
from os import listdir
from matplotlib import pyplot as plt


if __name__ == "__main__":
    theta = 15
    path = 'coin_ntd/'
    path_mask = 'coin_ntd_mask/'

    # list files in folder
    files = os.listdir(path)
    #files_mask = os.listdir(path_mask)

    for f in files:       
       # split name, extention
       f_name, f_ext = os.path.splitext(f)
       print (f_name, f_ext)

       # read image
       img = cv2.imread(path + f, 1)
       img_mask = cv2.imread(path_mask + f, 1)
       #print img.shape, img_mask.shape, img_mask[300,300]

       #cv2.imshow('image',img)
       #cv2.waitKey(0)

       #cv2.imshow('img_mask',img_mask)
       #cv2.waitKey(0)
       
       for h in range(img.shape[0]):
          for w in range(img.shape[1]):
             if not (np.array_equal(img_mask[h,w], (255,255,255))):
                img[h,w] = (255,255,255)
       #cv2.imshow("Contours", img)
       #cv2.waitKey(0)
       cv2.imwrite('coin_ntd_removebg/' + f_name + f_ext, img)
