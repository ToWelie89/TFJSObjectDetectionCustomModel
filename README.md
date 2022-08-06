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

1.1 Open up your Anaconda prompt and start it as administrator

![image](https://user-images.githubusercontent.com/5618925/183266926-5da658db-a195-40ea-a75e-fe56527da97e.png)


intall miniconda
create a conda enviroment
pip install tensorflow
conda activate myenv

## **Step 2 - Configure CUDA and cuDNN**

**Training a machine learning model is a very resource intensive process that ideally requires a strong computer. This is why many perform this operation on a remote machine that is specifically equipped to handle these demanding operations. However in this tutorial we will do everything on the local computer, without any cloud services. It is generally recommended to perform the training phase using a GPU (graphics card). If you do not have a dedicated graphics card on your computer (or simply feeling lazy and don't want to do this step) you can simply skip this step and go to [step 3](#step-3---start-creating-your-dataset), because it is possible to just use your CPU, but it is slower and not ideal. If you do have a GPU however you can follow the these instructions setup your computer so that it is ready to train models using a GPU.**

*Some information; CUDA (or Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs. cuDNN (CUDA Deep Neural Network library) on the other hand is a GPU-accelerated library of primitives for deep neural networks. To setup our environment we will need both CUDA and cuDNN.*

## **Step 3 - Start creating your dataset**

### Getting images
### Label images
### xml to csv

## **Step 4 - Prepare for training**

## **Step 5 - Perform the training**

## **Step 6 - Format your results**

## **Step 7 - Setup your Javascript environment**

## **Step 8 - Try it out**


