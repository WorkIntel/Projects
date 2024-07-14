# Intel RealSense Object Camera - DPC

Objects are everywhere!

This package supports object data extraction from the intel Realsense camera RGB or Depth streams in real time.

The list of the supported objects, frameworks, platforms and applications are below. Click on each image to find out more.

# Objects

-  Barcodes, QR Codes, Aruco Markers - well defined objects detected by using RGB data. 

Barcodes   | QR Codes | Aruco Markers |
:------------: |  :----------: | :-------------:  |
[![Barcode](https://github.com/WorkIntel/Projects/blob/main/Barcode/doc/barcode_camera-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Barcode/README.md)  | [![QR Codes](https://github.com/WorkIntel/Projects/blob/main/Barcode/doc/qrcode_camera-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Barcode/README.md)  | [![Aruco](Barcode/doc/aruco_camera-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Barcode/README.md)  |

-  Planes, Edges, Corners - 3D depth objects detected by using Depth data. 

Planes | Edges | Corners |
:------------: |  :----------: | :-------------:  |
[![Depth Sensing](https://user-images.githubusercontent.com/32394882/230639409-356b8dfa-df66-4bc2-84d8-a25fd0229779.gif)](https://www.stereolabs.com/docs/depth-sensing)  | [![Object Detection](https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif)](https://www.stereolabs.com/docs/object-detection)  | [![Yolo](https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif)](https://www.stereolabs.com/docs/body-tracking)  |

-  General Object Detection - well defined objects detected by using Depth data. 

Motion Detection/Safety | Object Detection in 2D | 2D Objects using YOLO |
:------------: |  :----------: | :-------------:  |
[![Safety](https://github.com/WorkIntel/Projects/blob/main/Safety/doc/motion_detection-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Safety/README.md)  | [![Object Detection](https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif)](https://www.stereolabs.com/docs/object-detection)  | [![Yolo](https://github.com/WorkIntel/Projects/blob/main/Yolo/doc/object_counting_output-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Yolo)  |

-  3D Pose estimation from RGB and depth data. 

3D Pose Estimation | Object Detection in 2D | Body Tracking |
:------------: |  :----------: | :-------------:  |
[![Pose6D](https://github.com/WorkIntel/Projects/blob/main/Pose6D/doc/pose6d-ezgif.com-video-to-gif-converter.gif)](https://github.com/WorkIntel/Projects/blob/main/Pose6D/README.md)  | [![Object Detection](https://user-images.githubusercontent.com/32394882/230630901-9d53502a-f3f9-45b6-bf57-027148bb18ad.gif)](https://www.stereolabs.com/docs/object-detection)  | [![Body Tracking](https://user-images.githubusercontent.com/32394882/230631989-24dd2b58-2c85-451b-a4ed-558d74d1b922.gif)](https://www.stereolabs.com/docs/body-tracking)  |


# Applications

User level applications supported by the Camera software

Region Detection | Object Counting | Object Tracking |
:------------: |  :----------: | :-------------:  |
[![Positional Tracking](https://user-images.githubusercontent.com/32394882/229093429-a445e8ae-7109-4995-bc1d-6a27a61bdb60.gif)](https://www.stereolabs.com/docs/positional-tracking/) | [![Global Localization](https://user-images.githubusercontent.com/32394882/230602944-ed61e6dd-e485-4911-8a4c-d6c9e4fab0fd.gif)](/global%20localization) | [![Spatial Mapping](https://user-images.githubusercontent.com/32394882/229099549-63ca7832-b7a2-42eb-9971-c1635d205b0c.gif)](https://www.stereolabs.com/docs/spatial-mapping) |

VSLAM/Localization | Plane Detection | Fall Detection |
:------------: |  :----------: | :-------------:  |
[![Camera Control](https://user-images.githubusercontent.com/32394882/230602616-6b57c351-09c4-4aba-bdec-842afcc3b2ea.gif)](https://www.stereolabs.com/docs/video/camera-controls/) | [![Plane Detection](https://user-images.githubusercontent.com/32394882/229093072-d9d70e92-07d5-46cb-bde7-21f7c66fd6a1.gif)](https://www.stereolabs.com/docs/spatial-mapping/plane-detection/)  | [![Multi Camera Fusion](https://user-images.githubusercontent.com/32394882/228791106-a5f971d8-8d6f-483b-9f87-7f0f0025b8be.gif)](/fusion) |

# Request Camera Feature
If you want to run the application or object detection on the camera hardware - check this [link](https://docs.google.com/forms/d/e/1FAIpQLSdduDbnrRExDGFQqWAn8pX7jSr8KnwBmwuFOR9dgUabEp0F1A/viewform).

# Supported Platforms and Compute Environments

The following is the check list of supported environments and functionality:
- Windows
- Ubuntu
- Jetson (NVIDIA)
- Raspeberry PI
- RealSense AI Engine

# How to Contribute

We greatly appreciate contributions from the community, including examples, applications, and guides. If you'd like to contribute, please follow these guidelines:

1. **Create a pull request (PR)** with the title prefix `[RS]`, adding your new example folder to the `examples/` directory within the repository.

2. **Ensure your project adheres to the following standards:**
   - Makes use of the `vision` package.
   - Includes a `README.md` with clear instructions for setting up and running the example.
   - Avoids adding large files or dependencies unless they are absolutely necessary for the example.
   - Contributors should be willing to provide support for their examples and address related issues.



