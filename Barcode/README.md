![](data/show_examples.jpg)

[![image](https://img.shields.io/pypi/v/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)
[![image](https://anaconda.org/conda-forge/scikit-spatial/badges/version.svg)](https://anaconda.org/conda-forge/scikit-spatial)
[![image](https://img.shields.io/pypi/pyversions/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)
[![image](https://github.com/ajhynes7/scikit-spatial/actions/workflows/main.yml/badge.svg)](https://github.com/ajhynes7/scikit-spatial/actions/workflows/main.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ajhynes7/scikit-spatial/master.svg)](https://results.pre-commit.ci/latest/github/ajhynes7/scikit-spatial/master)
[![Documentation Status](https://readthedocs.org/projects/scikit-spatial/badge/?version=latest)](https://scikit-spatial.readthedocs.io/en/latest/?badge=latest)
[![image](https://codecov.io/gh/ajhynes7/scikit-spatial/branch/master/graph/badge.svg)](https://codecov.io/gh/ajhynes7/scikit-spatial)


# Introduction

This code provides examples of the using RealSense camera for detection of Image objects.
The object detection is based on RGB values of the camera.
The following objects are supported:

-   Barcodes, QR Codes : detected by using  external library
-   ArucoMarkers : using OpenCV

These objects could be integrated in your robotics and video processing pipe line.


# Modules

We are using 


## Installation Windows

1. install python and virtual environment:
2. 

## Usage

```py
>>> from skspatial.objects import Vector

>>> vector = Vector([2, 0, 0])

```