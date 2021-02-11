'''
 Copyright 2020 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

import numpy as np
import os
import random
import cv2

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split

import dobble_utils as db

dir = './dobble_dataset'
nrows = 224
ncols = 224
nchannels = 3



def datadownload():

    test_dir = dir+'/dobble_test02_cards'

    test_cards = db.capture_card_filenames(test_dir)
    random.shuffle(test_cards)

    test_X,test_y = db.read_and_process_image(test_cards,nrows,ncols)
    del test_cards

    ntest = len(test_y)
 
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    print("")
    print("TEST DATA SETS:")
    print("Shape of test data (X) is :", test_X.shape)
    print("Shape of test data (y) is :", test_y.shape)

    train_X,val_X,train_y,val_y = train_test_split(test_X,test_y, test_size=0.20, random_state=2)

    # Scale image data from range 0:255 to range 0:1.0
    # Also converts train & test data to float from uint8
    #train_X = (train_X/255.0).astype(np.float32)
    #val_X = (val_X/255.0).astype(np.float32)

    # one-hot encode the labels
    train_y = tf.keras.utils.to_categorical(train_y-1, num_classes=58)
    val_y = tf.keras.utils.to_categorical(val_y-1, num_classes=58)

    print("")
    print("TRAINING/VALIDATION DATA SETS:")
    print("Shape of training data (X) is :", train_X.shape)
    print("Shape of training data (y) is :", train_y.shape)
    print("Shape of validation data (X) is :", val_X.shape)
    print("Shape of validation data (y) is :", val_y.shape)
        
    return (train_X,train_y), (val_X,val_y)
