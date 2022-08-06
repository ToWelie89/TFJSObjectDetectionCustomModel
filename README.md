# How to train a custom Tensorflow model for object detection with a GPU and use it with Javascript

## **Introduction**

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in machine learning and developers easily build and deploy machine learning powered applications.

In this tutorial I will instruct you how to setup your environment from scratch so that you can create a custom object detection model, train it and then convert it into a format that you can use with plain Javascript.

## **Result**

TODO

## Prerequisites

Download and install Anaconda Python from [here](https://www.anaconda.com/products/distribution). Anaconda is a python runtime that allows you to create separate environments each with their own dependencies. Make sure you check this box to add Anaconda to your PATH variable.

![image](https://user-images.githubusercontent.com/5618925/183266721-55589cee-9e5b-4331-a1ff-6af87fb3b7d8.png)

have node and git installed
use a windows computer, however linux could also use this tutorial

## **Step 1 - Setup the environment**

Open up your Anaconda prompt and start it as administrator

![image](https://user-images.githubusercontent.com/5618925/183266926-5da658db-a195-40ea-a75e-fe56527da97e.png)

Create a conda environment, this will be like a separated container that contains its own dependencies. Give it name like "myenv"
```bash
conda create --name myenv
```
Now activate your environment

```bash
conda activate myenv
```
You are now inside the "myenv" environment. If you would happen to close your Anaconda prompt, next time you open it up you must remember to once again activate your myenv environment before you do anything else.

Now install tensorflow
```bash
pip install tensorflow
```

intall miniconda
create a conda enviroment
pip install tensorflow
conda activate myenv

## **Step 2 - Configure CUDA and cuDNN (optional but recommended)**

**Training a machine learning model is a very resource intensive process that ideally requires a strong computer. This is why many perform this operation on a remote machine that is specifically equipped to handle these demanding operations. However in this tutorial we will do everything on the local computer, without any cloud services. It is generally recommended to perform the training phase using a GPU (graphics card). This tutorial is targeted towards using an Nvidia graphics card, which is why I use CUDA and cuDNN, which are Nvidia technologies. AMD have launched an equivalent technology called GPUFORT, but I have no experience with this. If you do not have a dedicated Nvidia graphics card on your computer (or simply feeling lazy and don't want to do this step) you can simply skip this step and go to [step 3](#step-3---start-creating-your-dataset), because it is possible to just use your CPU, but it is slower and not ideal. If you do have an Nvidia GPU however you can follow the these instructions setup your computer so that it is ready to train models using an Nvidia GPU.**

*Some information; CUDA (or Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs. cuDNN (CUDA Deep Neural Network library) on the other hand is a GPU-accelerated library of primitives for deep neural networks. To setup our environment we will need both CUDA and cuDNN.*

Before you get started, make sure you have the latest drivers for your GPU. I like keeping my drivers up to date with [GeForce Experience](https://www.nvidia.com/sv-se/geforce/geforce-experience/)

First you must find out which version of tensorflow you are using. Run
```bash
pip list tensorflow
```
Look up tensorflow and see its version. For me it is 2.9.1. Then go to [this](https://www.tensorflow.org/install/source_windows#gpu) page. Here you can see which version of CUDA and cuDNN that are compatible with your Tensorflow version. I can see in this list that for Tensorflow version 2.9.x I need cuDNN 8.1+ and CUDA 11.2+.

<img src="https://user-images.githubusercontent.com/5618925/183267375-304c7564-6d35-48ae-9531-83a3d117de0c.png" width="600">

[Here](https://developer.nvidia.com/cuda-toolkit-archive) you can find all CUDA versions. Note that you may need to create a free Nvidia developer account in order to access this page.

<img src="https://user-images.githubusercontent.com/5618925/183267431-b65124c3-0526-4bff-bc3d-9a181f53f065.png" width="600">

Make sure to select your correct Windows version and choose installer type exe (local). Simply download the exe and install it on your system.

Then download your compatible cuDNN version from [here](https://developer.nvidia.com/rdp/cudnn-archive).

To install cuDNN, simply extract the contents of your cuDNN downloaded zip-file into the install folder of CUDA, which by default should be **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X**. When prompted, select to replace existing files.

![image](https://user-images.githubusercontent.com/5618925/183267604-eaf6d9c1-3c9e-408e-abb9-12d79284c2ad.png)


## **Step 3 - Start creating your dataset**

### Getting images
### Label images
### xml to csv

## **Step 4 - Prepare for training**

## **Step 5 - Perform the training**

## **Step 6 - Format your results**

## **Step 7 - Setup your Javascript environment**

## **Step 8 - Try it out**


