#By @Kevin Xu

#kevin28520@gmail.com

#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw

#

#The aim of this project is to use TensorFlow to process our own data.

#    - input_data.py:  read in data and generate batches

#    - model: build the model architecture

#    - training: train



# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.

# With current settings, 10000 traing steps needed 50 minutes on my laptop.




# data: cats vs. dogs from Kaggle

# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

# data size: ~540M



# How to run?

# 1. run the training.py once

# 2. call the run_training() in the console to train the model.



# Note: 

# it is suggested to restart your kenel to train the model multiple times 

#(in order to clear all the variables in the memory)

# Otherwise errors may occur: conv1/weights/biases already exist......

import tensorflow as tf

import numpy as np
import os
import cv2

def get_files(file_dir):
    #print ('get_files')
    coin_1=[]
    label_1=[]
    coin_5=[]
    label_5=[]
    coin_10=[]
    label_10=[]
    coin_50=[]
    label_50=[]

    for file in os.listdir(file_dir):
        #print (file)
        name = file.split(sep='_')
        #print (name)
        if name[0]=='1':
            coin_1.append(file_dir+file)
            label_1.append(0)
        if name[0]=='5':
            coin_5.append(file_dir+file)
            label_5.append(1)
        if name[0]=='10':
            coin_10.append(file_dir+file)
            label_10.append(2)
        if name[0]=='50':
            coin_50.append(file_dir+file)
            label_50.append(3)

    #print('There are %d $1\nThere are %d $5\nThere are %d $10\nThere are %d $50' %(len(coin_1), len(coin_5), len(coin_10), len(coin_50)))
    image_list=np.hstack((coin_1, coin_5, coin_10, coin_50))
    label_list=np.hstack((label_1, label_5, label_10, label_50))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

def get_next_batch(images, labels, image_W, image_H, batch_size, capacity, startIdx=0):
    #print ('get_batch')
    '''
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size= batch_size, num_threads= 64, capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
    '''
    batch_images, batch_labels = [], []   
    
    # random dataset
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    images = list(temp[:, 0])
    labels = list(temp[:, 1])
    labels = [int(i) for i in labels]
    
    # read batch size images
    for file in images[:batch_size]:
        img = cv2.imread(file, 1)
        img_resize = cv2.resize(img, (image_W, image_H)) 
        batch_images.append(img_resize) 
    
    # build batch size labels, [[1,0,0,0],[0,0,1,0],...]
    for i in labels[:batch_size]:
        label = np.zeros(4)
        label[i] = 1
        batch_labels.append(label)
        
    return batch_images, batch_labels