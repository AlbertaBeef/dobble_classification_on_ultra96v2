# Dobble Classification using Vitis AI and TensorFlow
-----------------------

## Introduction

Introduction
In [part one](https://www.hackster.io/aidventure/the-dobble-challenge-93d57c) of this tutorial, we learned about the math behind the card game Dobble and looked at the dataset.

In [part two](https://www.hackster.io/aidventure/training-the-dobble-challenge-568854), we created and augmented our Dobble Card dataset, trained a Dobble-playing machine learning model, and tested it on a real life game between human and machine!

In this project, [part three](https://www.hackster.io/aidventure/deploying-the-dobble-challenge-on-the-ultra96-v2-f5f78f), we will deploy the Dobble classifier on the Ultra96-V2 development board.

We will run the following steps:

0. Download the dobble dataset from kaggle
1. Training and evaluation of the Dobble network using TensorFlow Keras.
2. Conversion of the HDF5 format Keras checkpoint into a TensorFlow compatible checkpoint.
3. Removal of the training nodes and conversion of the graph variables to constants (..often referred to as 'freezing the graph').
4. Evaluation of the frozen model using the Dobble test dataset.
5. Quantization of the floating-point frozen model.
6. Evaluation of the quantized model using the Dobble test dataset.
7. Compilation of the quantized model to create the ```.xmodel``` file ready for execution on the DPU accelerator IP.
8. Download and execution of the application on an evaluation board.


## The Dobble Dataset

The Dobble dataset was created by the authors of this project, and have been made available on kaggle:

[Kaggle - Dobble Card Images](https://www.kaggle.com/grouby/dobble-card-images/data)


The dataset contains ~500 images for training and ~1200 images for testing.


A more detailed description of the dataset can be found on hackster.io:

[Hackster - The Dobble Challenge](https://www.hackster.io/aidventure/the-dobble-challenge-93d57c)

A second project describes how to synthetically augment the dataset and train a classification model:

[Hackster - Training the Dobble Challenge](https://www.hackster.io/aidventure/training-the-dobble-challenge-568854)


There are a total of 58 mutually exclusive classes (or labels), each corresponding to a unique card.



## Implementing the Design

This section will lead you through the steps necessary to run the design in hardware.

### Preparing the Host Machine and Target Board

The host machine has several requirements that need to be met before we begin. You will need:

  + An Ubuntu 16.04 or 18.04 x86 host machine with internet access to download files.

  + Optionally, a GPU card suitable for training (a trained checkpoint is provided for those who wish to skip the training step).

  + The environment is supposed to be ready at this step. If not, please follow the setup instructions provided in [Module_2](https://github.com/Xilinx/Vitis-In-Depth-Tutorial/tree/master/Machine_Learning/Introduction/03-Basic/Module_2) and in [Module_3](https://github.com/Xilinx/Vitis-In-Depth-Tutorial/tree/master/Machine_Learning/Introduction/03-Basic/Module_3).

This project will run on the Vitis-AI 1.3 pre-built SD card image for Ultra96-V2, which can be found here:

[Hackster - Vitis-AI 1.3 Flow for Avnet Platforms](https://www.hackster.io/AlbertaBeef/vitis-ai-1-3-flow-for-avnet-vitis-platforms-cd0c51)


### Downloading the Design and Setting up the Workspace

This repository should be downloaded to the host machine as a zip file and then unzipped to a folder, or cloned using the ``git clone`` command from a terminal.

Open a linux terminal, cd into the repository folder then into the 'files' folder. Start the Vitis AI docker - if you have a GPU in the host system, it is recommended that you use the GPU version of the docker container. If you intend running the model training, you will definitely need the GPU docker container. If you are going to skip the training phase, then the CPU docker container will be sufficient.

As part of the [Setting up the host](https://www.xilinx.com/html_docs/vitis_ai/1_2/jck1570690043273.html) procedure, you will have cloned or downloaded The Vitis AI repository to the host machine. In the Vitis AI folder of that repo there is a shell script called docker_run.sh that will launch the chosen docker container. Open a terminal on the host machine and cd into the enter the following commands (note: start *either* the GPU or the CPU docker container, but not both):


```shell
# navigate to the dobble tutorial folder
cd <path_to_dobble_design>/files

# to start the CPU docker
./docker_run.sh xilinx/vitis-ai:1.3.411
```

If you have a GPU, you may prefer to use the GPU docker (which you will have to build)

```shell
# to start GPU docker
./docker_run.sh xilinx/vitis-ai-gpu:latest
```

The docker container will start and you should see something like this in the terminal:


```shell
==========================================

__      ___ _   _                   _____
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |
  \ \/ / | | __| / __|______/ /\ \   | |
   \  /  | | |_| \__ \     / ____ \ _| |_
    \/   |_|\__|_|___/    /_/    \_\_____|

==========================================

Docker Image Version:  1.3.411
Build Date: 2020-12-25
VAI_ROOT: /opt/vitis_ai

For TensorFlow Workflows do:
     conda activate vitis-ai-tensorflow
For Caffe Workflows do:
     conda activate vitis-ai-caffe
For Neptune Workflows do:
     conda activate vitis-ai-neptune
For PyTorch Workflows do:
     conda activate vitis-ai-pytorch
For TensorFlow 2.3 Workflows do:
     conda activate vitis-ai-tensorflow2
For Darknet Optimizer Workflows do:
     conda activate vitis-ai-optimizer_darknet
For Caffe Optimizer Workflows do:
     conda activate vitis-ai-optimizer_caffe
For TensorFlow 1.15 Workflows do:
     conda activate vitis-ai-optimizer_tensorflow
For LSTM Workflows do:
     conda activate vitis-ai-lstm
Vitis-AI /workspace >
```

Now run the environment setup script:  `source ./0_setenv.sh`

This will set up all the environment variables (..mainly pointers to folder and files..) most of which you can edit as required. It will also create the folders for the logs and the trained keras checkpoint.

The 0_setenv.sh script also activates the 'vitis-ai-tensorflow' TensorFlow conda environment, so you should now see that the terminal prompt looks like this:


```shell
(vitis-ai-tensorflow) Vitis-AI /workspace$
```

### Step 0: Download the Dobble dataset from Kaggle

Download the dataset archive (dobble-card-images.zip) from Kaggle.

[Kaggle - Dobble Card Images](https://www.kaggle.com/grouby/dobble-card-images/data)

Extract the archive under the files directory:

```shell
<path_to_dobble_design>/files
```

Rename the directory and ensure the dobble dataset is located as follows:

```shell
<path_to_dobble_design>/files/dobble_dataset
<path_to_dobble_design>/files/dobble_dataset/dobble_deck01_cards_57
...
<path_to_dobble_design>/files/dobble_dataset/dobble_deck10_cards_55
<path_to_dobble_design>/files/dobble_dataset/dobble_test01_cards
<path_to_dobble_design>/files/dobble_dataset/dobble_test02_cards
...
```

### Step 1: Training Your Model

:pushpin: Training takes a considerable time, between 8-12 hours depending on the GPU. You can alternatively:

+ Skip the training phase altogether and use the pretrained Keras checkpoint available in keras_model.zip. In this case, you can copy the k_model.h5 file inside this zip archive to the ./files/build/keras_model folder. You can then skip the remaining parts of Step 1 and go directly to Step 2.


To run step 1: ``source ./1_train.sh``

The training process is performed by the dobble_tutorial.py Python script.

For more information on the training, please refer to the following hackster.io project:

[Hackster - Training the Dobble Challenge](https://www.hackster.io/aidventure/training-the-dobble-challenge-568854)


After training has finished, the trained Keras checkpoint will be found in the ./files/build/keras_model folder as an HDF5 file called k_model.h5.

*Note: Any error messages relating to CUPTI can be ignored.*


### Step 2: Converting the Keras HDF5 Checkpoint to a TensorFlow Frozen Graph

To run step 2: ``source ./2_keras2tf.sh``

The Vitis AI tools cannot operate directly on Keras checkpoints and require a TensorFlow compatible frozen graph as the input format. The 2_keras2tf.sh shell script will create the frozen graph in two steps:

1. The HDF5 file is converted to a TensorFlow checkpoint.
2. The TensorFlow checkpoint is converted to a 'frozen graph' in binary protobuf format.

The output .pb file is generally known as a 'frozen graph' since all variables are converted into constants and graph nodes associated with training such as the optimizer and loss functions are stripped out.

After this step is completed, there should be a protobuf file called 'frozen_graph.pb' in the ./files/build/freeze folder.


### Step 3: Evaluating the Frozen Graph

To run step 3: ``source ./3_eval_frozen.sh``

This is an optional step as the frozen graph is still in floating-point format and should give almost identical accuracy results as the evaluation done during the training phase (step 1). The 1.2K images from Dobble test02 images are passed through the frozen model and the accuracy is calculated.


### Step 4: Quantizing the Frozen Graph

To run step 4: ``source ./4_quant.sh``

The DPU accelerator IP executes all calculations in 8bit integer format, so we must quantize our floating-point frozen graph. This is done by the Vitis AI tools, in particular by the 'vai_q_tensorflow quantize' command. This command can be seen in the 4_quant.sh script and has several arguments that we must provide values for:


| Argument               | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `--input_frozen_graph` | path and name of the input  .pb frozen graph                   |
| `--input_fn`           | Name of input function used in calibration pre-processing      |
| `--output_dir`         | Name of the output folder where the quantized models are saved |
| `--input_nodes`        | Name(s) of the input nodes                                     |
| `--output_nodes`       | Name(s) of the output nodes                                    |
| `--input_shapes`       | Shape(s) of the input nodes                                    |
| `--calib_iter`         | Number of calibration iterations                               |


*Note: Any error messages relating to ./bin/ptxas can be ignored.*

Most of the arguments are self-explanatory but special mention needs to be made of --input_fn and --calib_iter.

We require a sample set of data to calibrate the quantization process. This data will be passed through the model in one forward pass and so must be processed in exactly the same way as the data is pre-processed in training...the function pointed to be the --input_fn argument will need to contain all of the pre-processing steps.

The images for the calibration are created by the tf_gen_images.py python script and then stored in the ./files/build/quantize/images folder along with a text file which lists those images. This folder will be deleted after quantization is finished to reduce occupied disk space.

The image_input_fn.py Python script contains a single function called calib_input (..hence we set --input_fn to image_input_fn.calib_input in the 4_quant.sh shell script..) which opens the images with OpenCV, and then normalizes them to have all pixels in the range 0 to 1.0, exactly as was done in training and evaluation.

The number of images generated for use in calibration is set by the CALIB_IMAGES environment variable in the 0_setenv.sh script. Care should be taken that the number of calibration iterations (--calib_iter) multiplied by the calibration batch size (set in the image_input_fn.py script) does not exceed the total number of available images (CALIB_IMAGES).

Once quantization has completed, we will have the quantized deployment model (deploy_model.pb) and the evaluation model (quantize_eval_model.pb) in the ./files/build/quantize folder.


### Step 5: Evaluating the Quantized Model

To run step 5: ``source ./5_eval_quant.sh``

This is an optional, but *highly* recommended step. The conversion from a floating-point model where the values can have a very wide dynamic range to an 8bit model where values can only have one of 256 values almost inevitably leads to a small loss of accuracy. We use the quantized evaluation model to see exactly how much impact the quantization has had.

The exact same Python script, eval_graph.py, that was used to evaluate the frozen graph is used to evaluate the quantized model.


### Step 6: Compiling the Quantized Model

To run step 6: ``source ./6_compile.sh``

The DPU IP is a soft-core IP whose only function is to accelerate the execution of convolutional neural networks. It is a co-processor which has its own instruction set - those instructions are passed to the DPU in Xmodel file format.

The Vitis AI compiler will convert, and optimize where possible, the quantized deployment model to a set of micro-instructions and then output them in an Xmodel file.

The generated instructions are specific to the particular configuration of the DPU. The DPU's parameters are contained in a arch.json file which needs to be created for each target board - see the [Vitis AI User Guide](https://www.xilinx.com/html_docs/vitis_ai/1_3/zmw1606771874842.html) for details.

In the specific case of the Ultra96-V2, the arch.json file is provided in the B2304_LR directory and its location is passed to the vai_c_tensorflow command via the --arch argument.

Once compile is complete, the Xmodel file will be stored in the ./build/compile folder.


### Step 7: Running the Application on the Board

To run step 7: ``source ./7_make_target.sh``


This final step will copy all the required files for running on the board into the ./build/target folder. The entire target folder will be copied to the Ultra96-V2's SD card.

Copy it to the /home/root folder of the flashed SD card, this can be done with ```scp``` command.

  + If the Ultra96-V2 is connected to the same network as the host machine, the target folder can be copied using scp.

  + The command will be something like ``scp -r ./target root@192.168.1.227:~/``  assuming that the Ultra96-V2 IP address is 192.168.1.227 - adjust this and the path to the target folder as appropriate for your system.

  + If the password is asked for, insert 'root'.


With the target folder copied to the SD Card and the Ultra96-V2 booted, you can issue the command for launching the application - note that this done on the Ultra96-V2 board, not the host machine, so it requires a connection to the Ultra96-V2 such as a serial connection to the UART or an SSH connection via Ethernet.

By default, the serial console will output a significant quantity of verbose messages related to the Vitis-AI runtime.  This can be disabled with the following command:

```shell
root@u96v2-sbc-base-2020-2:~# dmesg -D
```

It is recommended to run the DPU optimization script, and set the monitor resolution (if connected) to 640x480:

```shell
root@u96v2-sbc-base-2020-2:~# cd dpu_sw_optimize/zynqmp
root@u96v2-sbc-base-2020-2:~/dpu_sw_optimize/zynqmp# ./zynqmp_dpu_optimize.sh

root@u96v2-sbc-base-2020-2:~# export DISPLAY=:0.0
root@u96v2-sbc-base-2020-2:~# xrandr --output DP-1 --mode 640x480
```


The application can be started by navigating into the target folder and then issuing the command ``python3 app_mt.py``. The application will start and after a few seconds will show the throughput in frames/sec.

```shell
root@u96v2-sbc-base-2020-2:~/target# python3 app_mt.py
Command line options:
 --image_dir :  images
 --threads   :  1
 --model     :  model_dir/dobble.xmodel
Pre-processing 500 images...
Starting 1 threads...
FPS=75,51, total frames = 500 , time=6.6213 seconds
output buffer length: 500
Correct: 489 Wrong: 11 Accuracy: 0.978
```

For better throughput, the number of threads can be increased like this:

```shell
root@u96v2-sbc-base-2020-2:~/target# python3 app_mt.py -t 5
Command line options:
 --image_dir :  images
 --threads   :  5
 --model     :  model_dir/dobble.xmodel
Pre-processing 500 images...
Starting 5 threads...
FPS=98.38, total frames = 500 , time=5.0821 seconds
output buffer length: 500
Correct: 489 Wrong: 11 Accuracy: 0.978
```


## Accuracy & Performance Results

The floating-point post-training and frozen graph evaluations can be compared to the INT8 post-quantization model and actual results obtained by the hardware model running on the Ultra96-V2 board:


| Post-training (Float) | Frozen Graph (Float) | Quantized Model (INT8) | Hardware model (INT8) |
| :-------------------: | :------------------: | :--------------------: | :-------------------: |
|        99.23%         |        99.25%        |         99.58%         |        97.80%         |


The approximate throughput (in frames/sec) for various batch sizes is shown below:


| Threads | Throughput (fps) |
| :-----: | :--------------: |
|    1    |       72.43      |
|    2    |      101.26      |
|    3    |      100.74      |
|    4    |       97.45      |
|    5    |       98.38      |


## Acknowledgment

This implementation is based on the DenseNet example from Xilinx:

[Module 4 - CIFAR10 Classification using Vitis AI and TensorFlowDesign Tutorials](https://github.com/Xilinx/Vitis-In-Depth-Tutorial/tree/master/Machine_Learning/Introduction/03-Basic/Module_4)



# References

## Vitis-AI 1.3 Flow for Avnet Platforms
This guide provides detailed instructions for targeting the Xilinx Vitis-AI 1.3 flow for Avnet Vitis 2020.2 platforms.

[Hackster - Vitis-AI 1.3 Flow for Avnet Platforms](https://www.hackster.io/AlbertaBeef/vitis-ai-1-3-flow-for-avnet-vitis-platforms-cd0c51)

## The Dobble Dataset
The Dobble dataset, available on kaggle:

[Kaggle - Dobble Card Images](https://www.kaggle.com/grouby/dobble-card-images/data)

## The Dobble Challenge
Getting started with machine learning for the Dobble card game using TensorFlow.

[Hackster - The Dobble Challenge](https://www.hackster.io/aidventure/the-dobble-challenge-93d57c)

## Training The Dobble Challenge
Train a machine learning model that can play Dobble (Spot-It) against you. This is part two of The Dobble Challenge.

[Hackster - Training The Dobble Challenge](https://www.hackster.io/aidventure/training-the-dobble-challenge-568854)



<p align="center"><sup>Copyright&copy; 2021 Avnet</sup></p>
