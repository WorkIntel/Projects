![](doc/planes.png)

[![image](https://img.shields.io/pypi/v/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)


# Introduction

This code provides examples of the using RealSense camera for detection of RGB and Depth objects.
The following objects are supported:

-   Planes : detected multiple planes in the depth image
-   Edges  : detecting edges / intersection of planes 
-   Corners: 3 plane intersection/ junctions

These objects could be integrated in your robotics and video processing pipe line.


## Installation Windows

1. install python and virtual environment:
2. 

## Usage

```py
>>> from Planes import plane_detector

>>> pd = plane_detector()

```