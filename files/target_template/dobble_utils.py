#
# Dobble Buddy - Utilities
#
# References:
#   https://www.kaggle.com/grouby/dobble-card-images
#
# Dependencies:
#   numpy
#   cv2
#   os
#   gc
#   csv
#   collections


import numpy as np
import cv2

import os
import gc

import csv
from collections import OrderedDict

#
# Capture images/labels from data set for training and testing
#

def capture_card_filenames(directory_name):
    subdirs = ['{}/{}'.format(directory_name,i) for i in sorted(os.listdir(directory_name)) ]
    cards = []
    for i,subdir in enumerate(subdirs):
        cards += ['{}/{}'.format(subdir,i) for i in os.listdir(subdir)]
    del subdirs
    return cards


#
# Read images and pre-process to fixed size
#

def read_and_process_image(list_of_images,nrows,ncols):
    X = []
    y = []
    
    for i,image in enumerate(list_of_images):
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncols), interpolation=cv2.INTER_CUBIC))
        y_str = image.split('/')
        y.append(int(y_str[len(y_str)-2]))
    return X,y


#
# Create collage for cards 01-55 (5 rows, 11 columns)
#

def create_collage(deck_id,cards_X,cards_y):

    cards_idx = np.where(np.logical_and(cards_y>=1, cards_y<=55))
    cards_55 = cards_X[cards_idx]
    
    h,w,z = cards_X[0,:,:,:].shape
    w11 = w * 11
    h5 = h * 5
    collage = np.zeros((h5,w11,3),np.uint8)
    idx = 0
    for r in range(0,5):
        for c in range(0,11):
            collage[r*h:(r+1)*h,c*w:(c+1)*w,:] = cards_55[idx,:,:,:]
            idx = idx + 1

    return collage


#
# Load Symbol labels and Card-Symbol mapping
#

def load_symbol_labels( symbol_filename ):
    symbols = OrderedDict()
    with open(symbol_filename,'r') as file:
        reader = csv.reader(file)
        symbol_id = 1
        for row in reader:
            symbol_label = row[1]
            symbols[symbol_id] = symbol_label
            #
            symbol_id = symbol_id + 1
    return symbols
    
#
# Load Card-Symbol mapping
#

def load_card_symbol_mapping( mapping_filename ):
    mapping = OrderedDict()
    with open(mapping_filename,'r') as file:
        reader = csv.reader(file)
        card_id = 1
        for row in reader:
            card_mapping = []
            for i,val in enumerate(row[1:]):
                if val=='1':
                    card_mapping.append( i+1 )
            mapping[card_id] = card_mapping
            #
            card_id = card_id + 1
    
    return mapping

