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

'''
Creates JPEG image files from Keras dataset in numpy array format
'''

import argparse
import os
import random
import cv2
import numpy as np

import dobble_utils as db

dir = './dobble_dataset'
nrows = 224
ncols = 224

#card_decks = [
#    'dobble_deck01_cards_57',
#    'dobble_deck02_cards_55',
#    'dobble_deck03_cards_55',
#    'dobble_deck04_cards_55',
#    'dobble_deck05_cards_55',
#    'dobble_deck06_cards_55',
#    'dobble_deck07_cards_55',
#    'dobble_deck08_cards_55',
#    'dobble_deck09_cards_55',
#    'dobble_deck10_cards_55'
#    ]
card_decks = [
    'dobble_test02_cards'
    ]
nb_card_decks = len(card_decks)

def gen_images(image_dir,calib_list,max_images,dataset):
    
    #
    # Capture images/labels from data set for training and testing
    #

    dobble_cards = []
    for d in range(0,nb_card_decks):
        card_dir = dir+'/'+card_decks[d]
        dobble_cards.append( db.capture_card_filenames(card_dir) )

    dobble_cards = np.concatenate( dobble_cards, axis=0 )
    random.shuffle(dobble_cards)

    # create file for list of calibration images
    # folder specified by args.calib_dir must exist
    if not (calib_list==''):
        f = open(os.path.join(image_dir, calib_list), 'w')

    for i in range(0,len(dobble_cards)):
        filename = dobble_cards[i]
        filename_split = filename.split('/')
        deck = filename_split[len(filename_split)-3]
        #card = filename_split[len(filename_split)-2]
        path,card = os.path.split(filename)
        im = cv2.imread(filename)

        filename = os.path.join(image_dir, deck + "_" + card)
        cv2.imwrite(filename,im)
        if not (calib_list==''):
            f.write(filename+'\n')

    if not (calib_list==''):
        f.close()

    print('Calib images generated')
    return


# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-dir', '--image_dir',
                  type=str,
                  default='images',
                  help='Path to folder for saving images and images list file. Default is images')  
  ap.add_argument('-l', '--calib_list',
                  type=str,
                  default='',
                  help='Name of images list file. Default is empty string so that no file is generated.')  
  ap.add_argument('-m', '--max_images',
                  type=int,
                  default=1,
                  help='Number of images to generate. Default is 1')
  ap.add_argument('-d', '--dataset',
                  type=str,
                  default='train',
                  choices=['train','test'],
                  help='Use train or test dataset. Default is train')
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' --image_dir    : ', args.image_dir)
  print (' --calib_list   : ', args.calib_list)
  print (' --dataset      : ', args.dataset)
  print (' --max_images   : ', args.max_images)

  gen_images(args.image_dir,args.calib_list,args.max_images,args.dataset)


if __name__ == '__main__':
  main()



