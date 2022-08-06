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

## **Step 2 - Configure CUDA and cuDNN (optional but recommended)**

**Training a machine learning model is a very resource intensive process that ideally requires a strong computer. This is why many perform this operation on a remote machine that is specifically equipped to handle these demanding operations. However in this tutorial we will do everything on the local computer, without any cloud services. It is generally recommended to perform the training phase using a GPU (graphics card). This tutorial is targeted towards using an Nvidia graphics card, which is why I use CUDA and cuDNN, which are Nvidia technologies. AMD have launched an equivalent technology called GPUFORT, but I have no experience with this. If you do not have a dedicated Nvidia graphics card on your computer (or simply feeling lazy and don't want to do this step) you can simply skip this step and go to [step 3](#step-3---start-creating-your-dataset), because it is possible to just use your CPU, but it is slower and not ideal. If you do have an Nvidia GPU however you can follow the these instructions setup your computer so that it is ready to train models using an Nvidia GPU.**

*Some information; CUDA (or Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs. cuDNN (CUDA Deep Neural Network library) on the other hand is a GPU-accelerated library of primitives for deep neural networks. To setup our environment we will need both CUDA and cuDNN.*

First make sure you have the latest drivers for your GPU. I like keeping my drivers up to date with [GeForce Experience](https://www.nvidia.com/sv-se/geforce/geforce-experience/)

Secondly, Download and install the latest version of [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/). This is because VS will also install many necessary dependencies. During the install when asked which components you want to add, just continue with the intall. You don't need any of those components, you only need the core editor.

![image](https://user-images.githubusercontent.com/5618925/183268745-dd264c0f-a2d1-41dc-9a6c-8aeba0079b91.png)

First you must find out which version of tensorflow you are using. Run
```bash
pip list tensorflow
```
Look up tensorflow and see its version. For me it is 2.9.1. Then go to [this](https://www.tensorflow.org/install/source_windows#gpu) page. Here you can see which version of CUDA and cuDNN that are compatible with your Tensorflow version. I can see in this list that for Tensorflow version 2.9.x I need cuDNN 8.1+ and CUDA 11.2+.

<img src="https://user-images.githubusercontent.com/5618925/183267375-304c7564-6d35-48ae-9531-83a3d117de0c.png" width="600">

[Here](https://developer.nvidia.com/cuda-toolkit-archive) you can find all CUDA versions.

<img src="https://user-images.githubusercontent.com/5618925/183267431-b65124c3-0526-4bff-bc3d-9a181f53f065.png" width="600">

Make sure to select your correct Windows version and choose installer type exe (local). Simply download the exe and install it on your system.

Then download your compatible cuDNN version from [here](https://developer.nvidia.com/rdp/cudnn-archive). To download cuDNN you will be asked to create an Nvidia account. Just make one, it is free.

To install cuDNN, simply extract the contents of your cuDNN downloaded zip-file into the install folder of CUDA, which by default should be **C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X**. When prompted, select to replace existing files.

![image](https://user-images.githubusercontent.com/5618925/183267604-eaf6d9c1-3c9e-408e-abb9-12d79284c2ad.png)

Now add the following folder paths to you PATH environment variable

1. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
2. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\libnvvp

Simply adjust these paths so they correspond with YOUR CUDA-version and install folder location. If you don't know how to edit your PATH environment variable you can read [this](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/).

Lastly open up your Anaconda prompt, make sure you are still in the myenv environment, and then run this command (replace PATH with your folder path to CUDA, like C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7):

```bash
conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir="PATH"
```
Example:
```bash
conda env config vars set XLA_FLAGS=--xla_gpu_cuda_data_dir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
```

Close your Anaconda prompt and open it up again as administrator. This is because you need to restart your runtime in order to get all of the new environment variables that comes from the CUDA installation.

Remember to activate your myenv environment.
```bash
conda activate myenv
```
Create a file called test.py and copy the following contents into this file:
```python
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="valid", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10),
    ]
)
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
model = my_model()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
```
Now run it
```bash
python test.py
```

You don't need to worry about what this file does, this is simply to check that tensorflow correctly recognizes and uses CUDA.

What you want to check for is that it recognizes your GPU

![image](https://user-images.githubusercontent.com/5618925/183269238-a22e8e77-80a8-4922-8e29-d695134aa4ba.png)

and that cuDNN is loaded

![image](https://user-images.githubusercontent.com/5618925/183269222-ac9d6083-18f6-47c1-b290-16f3a1e0b902.png)

This means CUDA and cuDNN is configured correctly. Another easy way to check is to simply watch your GPU load in the Performance tab in Task Manager, it should be close to 100%.

![image](https://user-images.githubusercontent.com/5618925/183269482-cbbe8cae-5444-419d-9b2d-7625487a0485.png)

You may now close the script with CTRL+C.

## **Step 3 - Start creating your dataset**

### Getting images
### Label images
### xml to csv

## **Step 4 - Prepare for training**

## **Step 5 - Perform the training**

## **Step 6 - Format your results**

## **Step 7 - Setup your Javascript environment**

## **Step 8 - Try it out**


