﻿<!--
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

Author: Mario Bergeron, Avnet Inc

Based on: https://github.com/Xilinx/Vitis-AI-Tutorials/tree/master/Design_Tutorials/08-tf2_flow
  Original Author: Mark Harvey, Xilinx Inc
-->
<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI Tutorials</h1>
  </td>
 </tr>
 <tr>
 <td align="center"><h1>TensorFlow2 and Vitis AI design flow</h1>
 </td>
 </tr>
</table>

This tutorial shows you how to compile and run the same identical design and application code on a number of different Xilinx cards. The virtually seamless transition between Edge and Cloud is made possible by the Vitis&trade; AI RunTime (VART) which is common to all target platforms and its unified APIs.


### Current status

+ Tested on ZU1 Board, Ultra96-V2, and UltraZed-EV
+ Tools used: TensorFlow2.6 & Vitis AI 2.0


## Introduction

We will run the following steps:

  + Download and preparation of the Dobble dataset. Conversion of images to TFRecords.
  + Training and evaluation of a custom CNN using TensorFlow's built-in version of Keras.
  + Quantization of the floating-point model using the Xilinx® quantizer provided as part of Vitis AI.
  + Evaluation of the quantized model using the dogs-vs-cats test dataset.
  + Compilation of the quantized model for execution on the target boards.
  + Execution of the network on the target boards with the provided Python scripts.

The complete flow and the tools used at each step is shown in the figure below:

![Complete flow](files/img/fig1.png)



## The Kaggle Dobble Dataset

The [Kaggle Dobble dataset](https://www.kaggle.com/datasets/grouby/dobble-card-images) consists of 500+ images of varying dimensions, divided into the 58 classes, 55 of which are actual playing cards. Each image is intrinsically labelled or classified by its filename, for example the image with filename *02.deck01_card02_01.jpg* is obviously of class *2*.

The 500+ images are all resized to 224 x 224 pixels and then divided into one of the train, validation or test datasets. The size of 224 x 224 pixels was chosen as a typical classification image size.



## The Convolutional Neural Network

The customcnn.py script uses the Keras Functional API to describe the simple CNN. It is a fully convolutional network and has no fully connected or dense layers. There are also no pooling layers — data reduction is achieved by using convolutional layers that have strides greater than one.

The CNN has deliberately been kept simple (so the expected prediction accuracy will not be much higher than approximately 95%. To reduce overfitting, batch normalization, dropout and L2 kernel regularization have been used.

![Custom CNN architecture](files/img/fig2.png)



The number of skip blocks and the number of filters used in each one is set by the 'filters' list argument - one skip block will be created for each element in the list.


## Before You Begin

The host machine has several requirements that need to be met before we begin. You will need:

  + An x86 host machine with a supported OS and either the CPU or GPU versions of the Vitis-AI docker installed - see [System Requirements](https://github.com/Xilinx/Vitis-AI/blob/master/docs/learn/system_requirements.md).

  + The host machine will require Docker to be installed and the Vitis-AI CPU or GPU docker image to be built - see [Getting Started](https://github.com/Xilinx/Vitis-AI#getting-started).

  + A GPU card suitable for training.

  + If you plan to use the Ultra96-V2 development board, it should be prepared with the board image as per the [Vitis-AI 2.0 Flow for Avnet Vitis Platforms](https://avnet.me/vitis-ai-2.0-project) instructions.

For more details, refer to the latest version of the *Vitis AI User Guide* ([UG1414](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)).

This tutorial assumes the user is familiar with Python3, TensorFlow and has some knowledge of machine learning principles.



## Setting up the workspace and dataset

1. Copy the repository by doing either of the following:

    + Download the repository as a ZIP file to the host machine, and then unzip the archive.
    + From a terminal, use the `git clone` command.

2. Download the Kaggle Dobble dataset.

    + Go to the [Kaggle website](https://www.kaggle.com/datasets/grouby/dobble-card-images) and register a new account if necessary.
    + Download the [dataset](https://www.kaggle.com/datasets/grouby/dobble-card-images).
    + Move dobble-card-images.zip into the `files` folder in the design repository, which is the same folder that contains the python (`.py`) and shell (`.sh`) scripts.

    The Kaggle Dobble dataset consists of 500+ images of varying dimensions, divided into 58 classes. Each image is intrinsically labelled or classified by its filename (for example, `01.tif`).

    There is a set of unlabelled images which were part of the original Kaggle dogs-vs-cats challenge, but we will not use it in this tutorial. Only the 500+ images that are contained in the `train.zip` archive will be used.

3. Open a linux terminal, `cd` to the repository folder, and then `cd` to the `files` folder.
   
4. Start the Vitis AI GPU docker:

     ```shell
     # navigate to tutorial folder
     cd <path_to_tutorial>/files

     # to start GPU docker container
     ./docker_run.sh xilinx/vitis-ai-gpu:latest
     ```

  The docker container will start and after accepting the license agreement, you should see something like this in the terminal:

     ```shell
     ==========================================
      
     __      ___ _   _                   _____
     \ \    / (_) | (_)            /\   |_   _|
      \ \  / / _| |_ _ ___ ______ /  \    | |
       \ \/ / | | __| / __|______/ /\ \   | |
        \  /  | | |_| \__ \     / ____ \ _| |_
         \/   |_|\__|_|___/    /_/    \_\_____|
      
     ==========================================
     
     Docker Image Version: 2.0.0.1103   (CPU) 
     Vitis AI Git Hash: 06d7cbb 
     Build Date: 2022-01-12
     
     For TensorFlow 1.15 Workflows do:
          conda activate vitis-ai-tensorflow 
     For Caffe Workflows do:
          conda activate vitis-ai-caffe 
     For PyTorch Workflows do:
          conda activate vitis-ai-pytorch 
     For TensorFlow 2.6 Workflows do:
          conda activate vitis-ai-tensorflow2 
     Vitis-AI /workspace > 
     ```

>:bulb: *If you get a "Permission Denied" error when starting the docker container, it is almost certainly because the docker_run.sh script is not set to be executable. You can fix this by running the following command:*
>    
>    ```shell
>     chmod +x docker_run.sh
>    ```


Activate the Tensorflow2 python virtual environment with `conda activate vitis-ai-tensorflow2` and you should see the prompt change to indicate that the environment is active:


```shell
Vitis-AI /workspace > conda activate vitis-ai-tensorflow2
(vitis-ai-tensorflow2) Vitis-AI /workspace > 
```

*The remainder of this README describes each single step to implement the tutorial, however a shell script called run_all.sh is provided which will run the complete flow:*

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > source run_all.sh
```

## Step 0 - Converting the dataset images to TFRecords

To run step 0:

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > python -u images_to_tfrec.py 2>&1 | tee tfrec.log
```

To speed up training, the JPEG images of the dogs-vs-cats dataset will be converted into the TFRecord format. The `images_to_tfrec.py` script will do the following:

+ Unzip the dogs-vs-cats.zip archive into the folder set by the `--dataset_dir` argument.
+ Split the images into the train and test datasets, ensuring a balance between classes.
+ Convert each image and label into a TFRecord. The TFRecord files are written into .tfrecord files in the folder deefined by the `--tfrec_dir` argument.
+ Move the test images to a separate folder for later use on the target.

Each TFRecord has five fields that are defined by the feature dictionary:

```python
# features dictionary
feature_dict = {
  'label' : _int64_feature(label),
  'height': _int64_feature(image_shape[0]),
  'width' : _int64_feature(image_shape[1]),
  'chans' : _int64_feature(image_shape[2]),
  'image' : _bytes_feature(image)
}
```

The label is obtained by looking at the first part of the image file name and assigning either '0' for dog or '1' for cat:

```python
class_name,_ = img.split('.',1)
if class_name == 'dog':
  label = 0
else:
  label = 1
```

Each JPEG image file is read into a TensorFlow string (tf.string) and its shape is obtained from the JPEG header - this avoids having to JPEG decode the image which means the script runs faster and also the TFRecord files are more compact:

```python
# read the JPEG source file into a tf.string
image = tf.io.read_file(filePath)

# get the shape of the image from the JPEG file header
image_shape = tf.io.extract_jpeg_shape(image, output_type=tf.dtypes.int32, name=None)
```

The number of image/label pairs in each .tfrecord file is defined by the `--img_shard` argument.



## Step 1 - Training

To run step 1:

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > python -u train.py 2>&1 | tee train.log
```

During training, the TFRecord files are read into the tf.data pipeline by the `input_fn_trn` function defined in `dataset_utils.py`. This function finds all TFRecord files whose names match the pattern train_*.tfrecord and creates a tf.data.Dataset object. The function also includes all the image pre-processing (resizing and random cropping, augmentation and normalization):


```python
def input_fn_trn(tfrec_dir,batchsize,height,width):
    '''
    Dataset creation and augmentation for training
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/train_*.tfrecord'.format(tfrec_dir), shuffle=True)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize_crop(x,y,h=height,w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset
```


The validation phase uses the `input_fn_test` function which will make a dataset from all TFRecord files which match the glob pattern test_*.tfrecord. Note how there is no augmentation, only resizing and normalization and the dataset does not repeat:


```python
def input_fn_test(tfrec_dir,batchsize,height,width):
    '''
    Dataset creation and augmentation for test
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}/test_*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize_crop(x,y,h=height,w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
```

The complete list of command line arguments of `train.py` are as follows:

|Argument|Default|Description|
|:-------|:-----:|:----------|
|--input_height|200|Input images are resized to input_height x input_width|
|--input_width|250|Input images are resized to input_height x input_width|
|--input_chan|3|Number of channels in input image - leave at default|
|--tfrec_dir|tfrecords|Folder containing TFRecord files|
|--batchsize|50|Batchsize used in training and validation - adjust for memory capacity of your GPU(s)|
|--epochs|250|Number of training epochs|
|--learnrate|0.001|Initial learning rate for optimizer|
|--chkpt_dir|float_model|Folder where trained checkpoint will be written|
|--tboard|tb_logs|Folder where TensorBoard logs will be written|


## Step 2 - Quantization

To run step 2:

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > python -u quantize.py --evaluate 2>&1 | tee quantize.log
```

The Xilinx DPU family of ML accelerators execute CNN models that have their parameters in integer format so we must convert the trained, floating-point checkpoint into a fixed-point integer checkpoint - this process is known as quantization.

The `quantize.py` script will do the following:

+ Make a folder (default name is quant_model) to contain the quantized model in HDF5 format.
+ Create a tf.data.Dataset object using the `input_fn_quant` defined in `dataset_utils.py`
     + this tf.data.Dataset is used to provide images for calibration.
+ Run the quantization process using the Vitis-AI quantizer plug-in for TensorFlow2.
+ Save the quantized HDF5 model to the folder indicated by the `--quant_model` command line argument.
+ If the `--evaluate` command line argument is included, then the `quantize.py` script will evaluate the accuracy of the quantized model using the same test dataset that was used for validation during training.


## Step 3 - Compiling for the target

To run step 3, run the `source compile.sh` with one of the target boards as a command line argument, for example:

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > source compile.sh zcu102
```

The script also supports `zcu104`,`vck190` and `u50`, as a command line arguments to target other boards. The `compile.sh` shell script will compile the quantized model and create an .xmodel file which contains the instructions and data to be executed by the DPU.


## Step 4 - Running the application on the target

To prepare the images, xmodel and application code for copying to the selected target, run the following command:

```shell
(vitis-ai-tensorflow2) Vitis-AI /workspace > python -u target.py -m compiled_zcu102/customcnn.xmodel -t target_zcu102 2>&1 | tee target_zcu102.log
```

The `target.py` script will do the following:

+ Resize the test images and copy them to the target folder.
     + the number of images is set by the `--num_images` command line argument which defaults to 1000.
+ Copy the compiled model to the target folder.
+ Copy the Python application code to the target folder.


### ZCU102

The entire `target_zcu102` folder should be copied to the ZCU102. Copy it to the /home/root folder of the flashed SD card, this can be done in one of several ways:

1. Direct copy to SD Card:

  + If the host machine has an SD card slot, insert the flashed SD card and when it is recognised you will see two volumes, BOOT and ROOTFS. Navigate into the ROOTFS and then into the /home folder.  Make the ./root folder writeable by issuing the command ``sudo chmod -R 777 root`` and then copy the entire `target_zcu102` folder from the host machine into the /home/root folder of the SD card.

  + Unmount both the BOOT and ROOTFS volumes from the host machine and then eject the SD Card from the host machine.

2. With scp command:

  + If the target evaluation board is connected to the same network as the host machine, the `target_zcu102` folder can be copied using scp.

  + The command will be something like ``scp -r ./build/target_zcu102 root@192.168.1.227:~/.``  assuming that the target board IP address is 192.168.1.227 - adjust this as appropriate for your system.

  + If the password is asked for, insert 'root'.


With the `target_zcu102` folder copied to the SD Card and the evaluation board booted, you can issue the command for launching the application - note that this done on the target evaluation board, not the host machine, so it requires a connection to the board such as a serial connection to the UART or an SSH connection via Ethernet.

The application can be started by navigating into the `target_zcu102` folder on the evaluation board and then issuing the command ``python3 app_mt.py``. The application will start and after a few seconds will show the throughput in frames/sec, like this:



```shell
root@xilinx-zcu102-2021_1:~# cd target_zcu102/
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  customcnn.xmodel
------------------------------------
Pre-processing 1000 images...
Starting 1 threads...
------------------------------------
Throughput=377.79 fps, total frames = 1000, time=2.6470 seconds
Post-processing 1000 images..
Correct:960, Wrong:40, Accuracy:0.9600
------------------------------------
```


The throughput can be improved by increasing the number of threads with the `--threads` option:


```shell
root@xilinx-zcu102-2021_1:~/target_zcu102# python3 app_mt.py --threads 8
Command line options:
 --image_dir :  images
 --threads   :  8
 --model     :  customcnn.xmodel
------------------------------------
Pre-processing 1000 images...
Starting 8 threads...
------------------------------------
Throughput=1055.84 fps, total frames = 1000, time=0.9471 seconds
Post-processing 1000 images..
Correct:960, Wrong:40, Accuracy:0.9600
------------------------------------
```
### ZCU104

The procedure is identical to that described above for the ZCU102 board - just use either the `target_zcu104` folder instead of the `target_zcu102` folder.




</hr>
<p class="sphinxhide" align="center"><sup>Copyright&copy; 2022 Avnet</sup></p>
