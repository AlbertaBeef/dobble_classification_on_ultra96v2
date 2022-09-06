'''
 Copyright 2022 Avnet Inc.

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
Make the target folder
Copies images, application code and compiled xmodel to 'target'
'''

'''
Author: Mario Bergeron, Avnet Inc

Based on: https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/08-tf2_flow
  Original Author: Mark Harvey, Xilinx Inc
'''


import argparse
import os
import shutil
import sys
import cv2
from tqdm import tqdm


DIVIDER = '-----------------------------------------'



def make_target(target_dir,image_dir,input_height,input_width,num_images,app_dir,model):

    # remove any previous data
    shutil.rmtree(target_dir, ignore_errors=True)    
    os.makedirs(target_dir)
    os.makedirs(target_dir+'/images')   

    # make a list of images
    images = os.listdir(image_dir)

    # resize each image and save to target folder
    print('Resizing and copying',num_images,'images...')
    for i in tqdm(range(num_images)):
        image = cv2.imread(os.path.join(image_dir, images[i]))
        image = cv2.resize(image, (input_width, input_height),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(target_dir,'images',images[i]), image)

    # copy application code
    print('Copying application code from',app_dir,'...')
    shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

    # copy compiled model
    print('Copying compiled model from',model,'...')
    shutil.copy(model, target_dir)


    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--target_dir', type=str,  default='build/target', help='Full path of target folder. Default is build/target')
    ap.add_argument('-i', '--image_dir',  type=str,  default='build/dataset-augmented/test', help='Full path of images folder. Default is build/dataset/test')
    ap.add_argument('-ih','--input_height',type=int, default=224,  help='Input image height in pixels.')
    ap.add_argument('-iw','--input_width', type=int, default=224,  help='Input image width in pixels.')
    ap.add_argument('-n', '--num_images', type=int,  default=8000, help='Number of test images. Default is 1000')
    ap.add_argument('-a', '--app_dir',    type=str,  default='application', help='Full path of application code folder. Default is application')
    ap.add_argument('-m', '--model',      type=str,  default='build/compiled_model/customcnn.xmodel', help='Full path of compiled model.Default is build/compiled_model/customcnn.xmodel')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --target_dir   : ', args.target_dir)
    print (' --image_dir    : ', args.image_dir)
    print (' --input_height : ', args.input_height)
    print (' --input_width  : ', args.input_width)
    print (' --num_images   : ', args.num_images)
    print (' --app_dir      : ', args.app_dir)
    print (' --model        : ', args.model)
    print('------------------------------------\n')


    make_target(args.target_dir, args.image_dir, args.input_height, args.input_width, args.num_images, args.app_dir, args.model)


if __name__ ==  "__main__":
    main()
