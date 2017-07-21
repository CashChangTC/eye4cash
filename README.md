# eye4cash

## Introduction

Use TensorFlow with CNN to classfy the NTD coin

## Dataset

Coin classes = [NTD-1, NTD-5, NTD-10, NTD-50], 
NTD-1: 230, 
NTD-5: 230, 
NTD-10: 299, 
NTD-50: 253

## Train

1. Enter Goofy: 
   docker exec -it TF_G3_Cash bash

2. Enter to train file path:
   ex: cd /home/Dev/Cash/Hackathon/
   
3. Execute train file
   python train.py
   
## Generate Train Image

1. Get coin mask in the source image. [coinmask](https://github.com/CashChangTC/eye4cash/tree/master/gen_train_data)

2. Use mask to remove the background. [removebackground](https://github.com/CashChangTC/eye4cash/tree/master/gen_train_data)

3. Rotate the image. [rotateimage](https://github.com/CashChangTC/eye4cash/tree/master/gen_train_data)
   
## Reference

[aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py)

