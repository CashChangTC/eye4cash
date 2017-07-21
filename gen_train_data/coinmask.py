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

    # list files in folder
    files = os.listdir(path)

    for f in files:
       # split name, extention
       f_name, f_ext = os.path.splitext(f)
       print (f_name, f_ext)

       # read image
       img = cv2.imread(path + f, 1)
       #cv2.imshow('image',img)
       #cv2.waitKey(0)

       # converter to gray
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
       # find circle
       #circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
       #print (circles)

       # binary
       #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
       thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 21, 2)
       #cv2.imshow("thresh", thresh)
       #cv2.waitKey(0)

       # remove noise
       kernel = np.ones((3,3), np.uint8)
       thresh = cv2.erode(thresh, kernel, iterations = 1)
       #cv2.imshow("remove noise", thresh)
       #cv2.waitKey(0)

       # find contour
       contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
       
       # calculate contour area
       
       new_contours = []
       height, width, channels = img.shape
       img_area_thresh = height*width*0.9
       max_cnt = []
       max_area = 0
       for cnt in contours:
          area = cv2.contourArea(cnt)
          if (area < img_area_thresh):
             #new_contours.append(cnt)
             if (max_area < area):
                max_area = area
                max_cnt = cnt
       new_contours.append(max_cnt)
       
       # draw contour
       cv2.drawContours(img, new_contours, -1, (255, 255, 255), cv2.cv.CV_FILLED)
       #img = np.zeros((512,512,3),np.uint8)
       #for center in circles[0]:
       #   cv2.circle(img, (center[0], center[1]), center[2], (0, 255, 0), 1)
       
       # show image
       #cv2.imshow("Contours", img)
       #cv2.waitKey(0)
       kernel = np.ones((5,5), np.uint8)
       thresh = cv2.dilate(thresh, kernel, iterations = 1)
       cv2.imwrite('coin_ntd_mask/' + f_name + f_ext, img)
       
       '''
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
          scipy.misc.imsave('ntd_rotated_center_color/' + f_name + '_' + str(theta) + f_ext, img_resize)
          theta+=15
          '''
