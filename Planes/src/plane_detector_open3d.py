
'''

Multi planar plain detector using open3d
==========================================

Using depth image to compute depth planes locally.
Based on https://github.com/yuecideng/Multiple_Planes_Detection/tree/master


Usage:

Environemt : 

    PowerShell: C:\\Users\\udubin\\AppData\\Local\\Programs\\Python\\Python310\\python.exe -m venv C:\\Users\\udubin\\Documents\\Envs\\planes
    Cmd: C:\\Users\\udubin\\Documents\Envs\\planes\\Scripts\\activate.bat

Install : 

    pip install opencv-contrib-python
    pip install scipy
    pip install matplotlib
    pip install pyrealsense2-2.55.10.6089-cp310-cp310-win_amd64.whl
    pip install open3d

'''

import numpy as np
import open3d as o3d
import unittest
import random
import time

from utils import plog, PointGenerator,  RealSense


#%% Utils
def NumpyToPCD(xyz):
    """ convert numpy ndarray to open3D point cloud 

    Args:
        xyz (ndarray): 

    Returns:
        [open3d.geometry.PointCloud]: 
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def PCDToNumpy(pcd):
    """  convert open3D point cloud to numpy ndarray

    Args:
        pcd (open3d.geometry.PointCloud): 

    Returns:
        [ndarray]: 
    """

    return np.asarray(pcd.points)

def RemoveNoiseStatistical(pc, nb_neighbors=20, std_ratio=2.0):
    """ remove point clouds noise using statitical noise removal method

    Args:
        pc (ndarray): N x 3 point clouds
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.

    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = NumpyToPCD(pc)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return PCDToNumpy(cl)

def PlaneRegression(points, threshold=0.01, init_n=3, iter=1000):
    """ plane regression using ransac

    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iter (int, optional): number of iteration. Defaults to 1000.

    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd         = NumpyToPCD(points)
    w, index    = pcd.segment_plane(threshold, init_n, iter)

    return w, index

def DrawResult(points, colors):
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)
    pcd.colors  = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray): 
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(target, threshold=threshold, init_n=3, iter=iterations)
        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    print('Found %d planes' %count)
    return plane_list


#%% Main
class PlaneDetector:
    def __init__(self):

        self.point_gen  = PointGenerator()
        self.points       = None # point cloud

        #self.img3d      = None # contains x,y and depth plains
        #self.imgXYZ     = None  # comntains X,Y,Z information after depth image to XYZ transform
        #self.imgMask    = None  # which pixels belongs to which cluster

        # params
        self.MIN_SPLIT_SIZE  = 32
        self.MIN_STD_ERROR   = 0.01

        # help variable
        #self.ang_vec     = np.zeros((3,1))  # help variable

    def init_points(self, img_type = 1, roi_type = 0):
        "init point cloud"
        self.points = self.point_gen.init_point_cloud(img_type,roi_type)
        self.tprint('Point cloud %d' %img_type)
        return True

    def preprocess_points(self, points):
        "cleaning the point cloud" 
        # pre-processing
        #points = RemoveNan(points)
        #points = DownSample(points,voxel_size=0.003)
        points  = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)       
        return points        

    def fit_planes(self, points):
        "fitting plane usong open3d"

        #DrawPointCloud(points, color=(0.4, 0.4, 0.4))
        t0      = time.time()
        results = DetectMultiPlanes(points, min_ratio=0.05, threshold=0.1, iterations=2000)
        self.tprint('Time:', time.time() - t0)
        self.tprint('Planes: %s' %str(len(results)))

    def show_planes(self, results):
        "picture the results"
        planes = []
        colors = []
        for _, plane in results:

            r = random.random()
            g = random.random()
            b = random.random()

            color = np.zeros((plane.shape[0], plane.shape[1]))
            color[:, 0] = r
            color[:, 1] = g
            color[:, 2] = b

            planes.append(plane)
            colors.append(color)
        
        planes = np.concatenate(planes, axis=0)
        colors = np.concatenate(colors, axis=0)
        DrawResult(planes, colors)
        self.tprint('Show done')

    def compute_and_show(self):
        "computes the planes"
        if self.points is None:
            self.tprint('Init points first','W')
            return False
        
        points   = self.preprocess_points(self.points)
        results  = self.fit_planes(points)
        self.show_planes(results)
        return True
        

    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            plog.info(txt)
        elif level == "W":
            plog.warning(txt)
        elif level == "E":
            plog.error(txt)
        else: # level is some other object
            plog.info(txt, str(level))

# ----------------------
#%% Tests
class TestPlaneDetector(unittest.TestCase):
                     
    def test_fit_plane(self):
        "computes normal to the ROI"
        p       = PlaneDetector()
        ret     = p.init_points(1, 4)
        ret     = p.compute_and_show()
        self.assertTrue(ret)  


#%% Main
if __name__ == "__main__":


    #unittest.main()
    suite = unittest.TestSuite()

    suite.addTest(TestPlaneDetector("test_fit_plane")) # ok

    runner = unittest.TextTestRunner()
    runner.run(suite)