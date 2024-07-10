![](data/show_examples.jpg)

[![image](https://img.shields.io/pypi/v/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)



# Introduction

This module provides some useful utilities and functions for RealSense camera usage.
The object detection is based on RGB values of the camera.
The following functions are supported:

-   OpenCV like camera usage
-   RGB and depth information alignments
-   Saving and storage of the RGB, Depth and RGD data

These functions are reused in other RealSense object detection modules.


# Modules

- opencv_realsense_camera 


## Installation Windows

Python is a self contained development environment. We use PIP to manage the package installation.
You can use Conda, Miniconda or other package managers.

1. Install Python:
2. Create virtual environment:
2. Activate the virtual environment

## Usage

```py
>>> from opencv_realsense_camera import RealSense

>>> rs_cap = RealSense('rgb')

```