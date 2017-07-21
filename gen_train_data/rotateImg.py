import os
import argparse
import numpy as np
import scipy

from skimage import data, io
from skimage import exposure, img_as_uint, img_as_float, transform
from scipy import ndimage
from PIL import Image
from os import listdir
from matplotlib import pyplot as plt

def readImage(path):
    files = os.listdir(path)
    img_array = []
    for f in files:
       img_array.append(img = ndimage.imread(path + f, mode='RGB'))
    return img_array

if __name__ == "__main__":
    theta = 15
    path = 'coin_ntd_removebg/'

    # list files in folder
    files = os.listdir(path)

    for f in files:
       # split name, extention
       f_name, f_ext = os.path.splitext(f)
       print (f_name, f_ext)

       # read image
       img = ndimage.imread(path + f, mode='RGB')
       #print (img_bg_value)
       
       # get origin image background part
       img_bg = img[:10,:10]
       img_bg_large = scipy.misc.imresize(img_bg, [500,500])
       
       # rotate image
       theta = 15
       while (theta < 360):
          img_rotated = ndimage.rotate(img, theta, reshape=True)

          # 
          for i in range(img_rotated.shape[0]):
             for j in range(img_rotated.shape[1]):
                if (np.array_equal(img_rotated[i,j], [0,0,0])):
                   img_rotated[i,j] = img[0,0]
          #plt.imshow(img_rotated)
          #plt.show()
        
          # paste onto large image
          if (img_rotated.shape[0] < img_rotated.shape[1]):
             img_large = scipy.misc.imresize(img_bg, [img_rotated.shape[1]*2,img_rotated.shape[1]*2])
          else:
             img_large = scipy.misc.imresize(img_bg, [img_rotated.shape[0]*2,img_rotated.shape[0]*2])
          #print (img_large)
          #plt.imshow(img_large)
          #plt.show()

          ir_h_helf, ir_w_helf = img_rotated.shape[0]/2, img_rotated.shape[1]/2
          il_h_helf, il_w_helf = img_large.shape[0]/2, img_large.shape[1]/2
          il_crop_h_start = il_h_helf-ir_h_helf
          il_crop_w_start = il_w_helf-ir_w_helf
          #print (ir_h_helf, ir_w_helf)
          #print

          img_large[il_crop_h_start:il_crop_h_start+img_rotated.shape[0], \
                    il_crop_w_start:il_crop_w_start+img_rotated.shape[1]] = img_rotated

          # resize image
          img_resize = scipy.misc.imresize(img_large, [500,500])
          #print (img_resize.shape)
          #plt.imshow(img_resize, cmap='gray')
          #plt.show()

          # save image
          scipy.misc.imsave('ntd_rotated_center_color_2/' + f_name + '_' + str(theta) + f_ext, img_resize)
          theta+=15
