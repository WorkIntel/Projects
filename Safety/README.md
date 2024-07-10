![](data/show_examples.jpg)

[![image](https://img.shields.io/pypi/v/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)



# Introduction

This code provides examples of the using RealSense camera for monitoring and detection of objects entering a predefined zones.
The code can use both RGB and Depth information to estimate backgrounnd.
The following functions are supported:

-   Estimate Background : estimates background using multiple gaussian models
-   Detection  : estimates regions that will be detected as intrusions

Theis code could be integrated in your robotics and video processing pipe line.
 

## Installation Windows

1. install python and virtual environment:
2. using barcode virtual environment
3. 

## Usage

```py
>>> from Safety.safety_detector import background_estimator

>>> bge = background_estimator()

```

## Test

In your environment run:
python  safety_detector.py
