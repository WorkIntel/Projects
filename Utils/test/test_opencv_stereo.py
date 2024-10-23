import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

if True:
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 800, 1200)

    cv2.createTrackbar('numDisparities', 'disp', 1, 50, nothing)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    #cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    #cv2.createTrackbar('preFilterSize', 'disp', 2, 25, nothing)
    #cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
    #cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 0, 100, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 0, 200, nothing)

# Creating an object of StereoBM algorithm
# stereo = cv2.StereoBM_create()
stereo = cv2.StereoSGBM_create()


Left_nice = cv2.imread(r'C:\Data\Corr\l2_Infrared.png')
Right_nice = cv2.imread(r'C:\Data\Corr\r2_Infrared.png')
Left_nice = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
Right_nice = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)

if True:
    while True:
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        #preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        #preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        #preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        #textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        #stereo.setPreFilterType(preFilterType)
        #stereo.setPreFilterSize(preFilterSize)
        #stereo.setPreFilterCap(preFilterCap)
        #stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32
        disparity = disparity.astype(np.float32)
        print(disparity[10:15,10:15])

        # Scaling down the disparity values and normalizing them
        #disparity = disparity / 16.0
        disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        cv2.imshow("disp", disparity)
        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break