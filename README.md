## PoseNet Python

This repository contains a pure Python implementation (multi-pose only) of the Google TensorFlow.js Posenet model.

I first adapted the JS code more or less verbatim and found the performance was low so made some vectorized numpy/scipy version of a few key functions (named `_fast`).

Further optimization is possible. The MobileNet base models have a throughput of 200-300 fps on a GTX 1080 Ti+. The _fast_ post processing code limits this to about 80-100fps if all file io and drawing is removed from the loop. A Cython or pure C++ port would be ideal. 

### Install

A suitable Python environment with a recent version of Tensorflow is required. Development was done with Tensorflow 1.12.0 from Conda.

A conda environment with these packages should suffice: `conda install tensorflow-gpu scipy pyyaml opencv`


### Usage

There are two demo apps to try the PoseNet model. They are very basic and could be made more performant.

The first time these apps are run (or the library is used) model weights will be downloaded for the TFJS version and converted on the fly.

For both demos, the model can be specified by using its ordinal id (0-3) or integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the get_test_images.py script.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen.


### Credits

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML


### TODO (someday, maybe)
* More stringent verification of correctness against the original implementation
* Performance improvements (especially edge loops in 'decode.py')
* OpenGL rendering/drawing
* Comment interfaces, tensor dimensions, etc

