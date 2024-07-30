![](data/show_examples.jpg)

[![image](https://img.shields.io/pypi/v/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)
[![image](https://anaconda.org/conda-forge/scikit-spatial/badges/version.svg)](https://anaconda.org/conda-forge/scikit-spatial)
[![image](https://img.shields.io/pypi/pyversions/scikit-spatial.svg)](https://pypi.python.org/pypi/scikit-spatial)


# Introduction

This code provides examples of the using RealSense camera for detection of Image objects.
The object detection is based on RGB values of the camera.
The following objects are supported:

-   Barcodes, QR Codes : detected by using  external library
-   ArucoMarkers : using OpenCV

These objects could be integrated in your robotics and video processing pipe line.


# Modules

We are using pyzbar and opencv contributed 


## Installation Windows

1. Install python 3.10 from Python Release Python 3.10.0 | Python.org

2. Create virtual environment. In Windows PowerShell:
Python -m venv <your path>\Envs\barcode

3. Activate virtual environment. In Windows CMD shell:
C:\Users\udubin\Documents\Envs\barcode\Scripts\activate.bat

4. Installing realsense driver. For example, download pyrealsense2-2.55.10.6089-cp310-cp310-win_amd64.whl:
pip install pyrealsense2-2.55.10.6089-cp310-cp310-win_amd64.whl

5. Install opencv and numpy:
pip install opencv-contrib-python

6. Instrall PyZbar library:
pip install pyzbar

7. Install scipy:
python -m pip install scipy

8. Install matplotlib:
pip install matplotlib

## Usage

```py
>>> from skspatial.objects import Vector

>>> vector = Vector([2, 0, 0])

```

## Troublshooting

1. During PyZbar installation if the following rrror happens: 

FileNotFoundError: Could not find module '<your path>\Envs\barcode\lib\site-packages\pyzbar\libzbar-64.dll' (or one of its dependencies). Try using the full path with constructor syntax.

install vcredist_x64.exe From <https://www.microsoft.com/en-gb/download/details.aspx?id=40784> 
Download Visual C++ Redistributable Packages for Visual Studio 2013 from Official Microsoft Download Center