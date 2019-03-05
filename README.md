# U-Net based semantic segmentation of trees in suburbs -- using WorldView-2 and Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

The original data is a WorldView-2 remote sensing image of Fuyang.
image size: 10256*10256.
image resolution: 0.5m(That means a pixel in the image is 0.25m^2 on Earth).
bands: 8(Red, Green, Blue, Near Infrared I, Coast, Yellow, Red Edge, Near Infrared II).

### Data preprocessing

The original image is too big for any deep learing architecture, so preprocessing is required. Evenworth, there is no groundtruth, which means I need to label the whole image by myself. After labeling, the pixel of tree:background = 39:61, which is a bit unbalance.
preprocessing includes following steps:
1.Radiance Calibration
2.NNDiffuse Pan Sharpening
3.Quick Atmospheric Correction
4.Adding NDVI as 9th band
5.Split into training set(1181), validation set(200), test set(300), all in 256*256 size.

The above steps are done by IDL and python. And it appears step 2 and 3 are reversed, in later researching, QUAC need to be done first.

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 256*256 which represents mask that should be learned. Sigmoid activation(loss:binary_crossentropy) function
makes sure that mask pixels are in \[0, 1\] range, and it is used in binary classification.
When facing multiple classification problems,softmax(loss:categorical_crossentropy) is better.
Kernel_initializer: he_normal
Optimizer: Adam
Learning rate: 1e-4
batchsize: 32
lr_decay: 0.5(after 6 patience)

### Training

The model is trained for 1000 epochs, though it converged early at about 100 epochs.

---

### Environments

* Spyder(Python 3.5)
* Tensorflow(1.12.0)
* Keras(2.2.4)

### Results

Use the trained model to do segmentation on test images, here are some of the results.

![img/good_raw.tif](img/good_raw.tif)![img/good_gt.tif](img/good_gt.tif)![img/good_pre.tif](img/good_pre.tif)

![img/med_raw.tif](img/med_raw.tif)![img/med_gt.tif](img/med_gt.tif)![img/med_pre.tif](img/med_pre.tif)

![img/bad_raw.tif](img/bad_raw.tif)![img/bad_gt.tif](img/bad_gt.tif)![img/bad_pre.tif](img/bad_pre.tif)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
