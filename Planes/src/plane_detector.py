#!/usr/bin/env python

'''
Multi planar plain detector
==================

Using depth image to compute depth planes locally.


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 



'''

import numpy as np
import cv2 as cv
import unittest
from scipy.spatial.transform import Rotation as Rot
from scipy import interpolate 

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense

import logging as log
log.basicConfig(stream=sys.stdout, level=log.DEBUG, format='[%(asctime)s.%(msecs)03d] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',  datefmt="%M:%S")
log.getLogger('matplotlib.font_manager').disabled = True
log.getLogger('matplotlib').setLevel(log.WARNING)
log.getLogger('PIL').setLevel(log.WARNING)

import matplotlib.pyplot as plt

#%% Helpers
def draw_axis(img, rvec, tvec, cam_mtrx, cam_dist, len = 10):
    # unit is mm
    points          = np.float32([[len, 0, 0], [0, len, 0], [0, 0, len], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _   = cv.projectPoints(points, rvec, tvec, cam_mtrx, cam_dist)
    axisPoints      = axisPoints.squeeze().astype(np.int32)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# # Code from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# # Calculates rotation matrix to euler angles
# # The result is the same as MATLAB except the order
# # of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R) :

#     sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

#     singular = sy < 1e-6

#     if  not singular :
#         x = np.arctan2(R[2,1] , R[2,2])
#         y = np.arctan2(-R[2,0], sy)
#         z = np.arctan2(R[1,0], R[0,0])
#     else :
#         x = np.arctan2(-R[1,2], R[1,1])
#         y = np.arctan2(-R[2,0], sy)
#         z = 0

#     theta = np.rad2deg(np.array([x, y, z]))
#     return theta


# # Calculates Rotation Matrix given euler angles.
# def eulerAnglesToRotationMatrix(theta) :

#     theta = np.deg2rad(theta)
 
#     R_x = np.array([[1,         0,                  0                   ],
#                     [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
#                     [0,         np.sin(theta[0]), np.cos(theta[0])  ]
#                     ])
 
#     R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
#                     [0,                     1,      0                   ],
#                     [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
#                     ])
 
#     R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
#                     [np.sin(theta[2]),    np.cos(theta[2]),     0],
#                     [0,                     0,                      1]
#                     ])
 
#     R = np.dot(R_z, np.dot( R_y, R_x ))
 
#     return R

#%% Main
class PlaneMatcher:
    def __init__(self):

        self.frame_size = (640,480)
        self.img        = None
        self.cam_matrix = np.array([[1000,0,self.frame_size[0]/2],[0,1000,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
        self.cam_distort= np.array([0,0,0,0,0],dtype = np.float32)

        self.img3d      = None # contains x,y and depth plains
        self.imgXYZ     = None  # comntains X,Y,Z information after depth image to XYZ transform

        # params
        self.MIN_SPLIT_SIZE  = 32
        self.MIN_STD_ERROR   = 0.01

        # help variable
        self.ang_vec     = np.zeros((3,1))  # help variable


    def init_image(self, img_type = 1):
        # create some images for test
        w,h             = self.frame_size
        if img_type == 1: # /
            
            self.img        = np.tile(np.linspace(100, 300, w), (h,1))

        elif img_type == 2: # /|/

            self.img        = np.tile(np.linspace(100, 200, int(w/2)), (h,2))
         
        elif img_type == 3: # |_|

            self.img        = np.tile(np.linspace(100, 200, h).reshape((-1,1)), (1,w)) 
        
        elif img_type == 4: # /\

            self.img        = np.tile(np.hstack((np.linspace(300, 500, w>>1),np.linspace(500, 300, w>>1))), (h,1))        

        elif img_type == 5: # dome

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = (np.abs(x - w/2) + np.abs(y - h/2))/10 + 200 # less slope

        elif img_type == 6: # sphere

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = np.sqrt((x - w/2)**2 + (y - h/2)**2)/10 + 200 # less slope   

        elif img_type == 7: # stair

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = (np.sign(x - w/2) + np.sign(y - h/2))*5 + 200 # less slope                     

        elif img_type == 10: # flat

            self.img        = np.ones((h,w))*500             

        elif img_type == 11:
            "chess board"
            fname = r"C:\Users\udubin\Documents\Code\opencv-4x\samples\data\left04.jpg"
            self.img        = cv.imread(fname)

        elif img_type == 12:
            self.img = cv.imread('image_scl_001.png', cv.IMREAD_GRAYSCALE)
            self.img = cv.resize(self.img , dsize = self.frame_size) 
            
        elif img_type == 13:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_ddd_000.png", cv.IMREAD_GRAYSCALE)
            self.img = cv.resize(self.img , dsize = self.frame_size) 

        elif img_type == 14:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_ddd_001.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)     

        elif img_type == 15:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_ddd_002.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)     

        elif img_type == 16:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_ddd_003.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)   

        elif img_type == 17:
            self.img = cv.imread(r"C:\Data\Depth\Plane\floor_view_default_Depth_Depth.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)          

        elif img_type == 18:
            self.img = cv.imread(r"C:\Data\Depth\Plane\floor_view_default_corner2_Depth_Depth.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)                   

        elif img_type == 21:
            self.img = cv.imread(r"C:\Data\Depth\Plane\image_scl_000.png", cv.IMREAD_GRAYSCALE)  
            self.img = cv.resize(self.img , dsize = self.frame_size)                                     
            
        #self.img        = np.uint8(self.img)       
        return self.img
      
    def init_roi(self, test_type = 1):
        "load the test case"
        roi = [0,0,self.frame_size[0],self.frame_size[1]]
        if test_type == 1:
            roi = [310,230,330,250] # xlu, ylu, xrb, yrb
        elif test_type == 2:
            roi = [300,220,340,260] # xlu, ylu, xrb, yrb
        elif test_type == 3:
            roi = [280,200,360,280] # xlu, ylu, xrb, yrb            
        elif test_type == 4:
            roi = [220,140,420,340] # xlu, ylu, xrb, yrb      
        elif test_type == 4:
            roi = [200,120,440,360] # xlu, ylu, xrb, yrb            
        return roi  


    def init_img3d(self, img = None):
        "initializes xyz coordinates for each point"
        img     = self.img if img is None else img
        h,w     = img.shape[:2]
        x       = np.arange(w)
        y       = np.arange(h)
        x,y     = np.meshgrid(x,y)
        fx      = self.cam_matrix[0,0]
        fy      = self.cam_matrix[1,1]
        
        xy      = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        xy      = np.expand_dims(xy, axis=1).astype(np.float32)
        xy_undistorted = cv.undistortPoints(xy, self.cam_matrix, self.cam_distort)

        u       = xy_undistorted[:,0,0].reshape((h,w))
        v       = xy_undistorted[:,0,1].reshape((h,w))
        z3d     = img.astype(np.float32)
        x3d     = z3d.copy()
        y3d     = z3d.copy()

        ii        = np.logical_and(z3d> 1e-6 , np.isfinite(z3d))
        x3d[ii]   = u[ii]*z3d[ii] #/fx
        y3d[ii]   = v[ii]*z3d[ii] #/fy
        z3d[ii]   = z3d[ii]

        #self.img3d = np.stack((u/fx,v/fy,z3d), axis = 2)
        self.img3d = np.stack((u,v,z3d), axis = 2)
        return self.img3d
    
    def compute_img3d(self, img = None):
        "compute xyz coordinates for each point using prvious init"
        img         = self.img if img is None else img
        xyz         = self.img3d
        if xyz is None:
            xyz = self.init_img3d(img)

        if np.any(img.shape[:2] != xyz.shape[:2]):
            print('Image dimension change')
            return 

        imgXYZ      = self.img3d.copy()

        z3d         = img.astype(np.float32)
        x3d         = self.img3d[:,:,0].copy()  # u/f
        y3d         = self.img3d[:,:,1].copy()  # v/f

        ii          = np.logical_and(z3d > 1e-6 , np.isfinite(z3d))
        x3d[ii]     = x3d[ii]*z3d[ii]
        y3d[ii]     = y3d[ii]*z3d[ii]
        z3d[ii]     = z3d[ii]

        # x,y,z coordinates in 3D
        imgXYZ[:,:,0] = x3d
        imgXYZ[:,:,1] = y3d
        imgXYZ[:,:,2] = z3d

        self.imgXYZ = imgXYZ
        return imgXYZ

    def detect_pose_in_chessboard(self): 
        # chess board pose extimation
        criteria    = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp        = np.zeros((6*7,3), np.float32)
        objp[:,:2]  = np.mgrid[0:7,0:6].T.reshape(-1,2)
        axis        = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        if len(self.img.shape) > 2:
            gray        = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        else:
            gray        = self.img
        flags_cv = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK #+ cv.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv.findChessboardCorners(gray, (7,6), flags = flags_cv)
        #ret, corners = cv.findChessboardCornersSB(gray, (7,6))
        if ret == True:
            corners = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        else:
            print('Failed to find points')
            return np.zeros((1,7))
            # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners, self.cam_matrix, self.cam_distort)
        # transfer to pose
        R, _       = cv.Rodrigues(rvecs)
        avec       = rotationMatrixToEulerAngles(R)
        pose       = np.hstack((tvecs.flatten(), avec.flatten(), 3))
        print('Chess pose : ', pose)
        return [pose]
    
    def check_error(self, xyz1_mtrx, vnorm):
        "checking the error norm"
        err         = np.dot(xyz1_mtrx, vnorm)
        err_std     = err.std()
        return err_std
    
    def fit_plane(self, roi):
        "computes normal for the specifric roi and evaluates error"
        x0,y0,x1,y1 = roi

        # reduce size of the grid for speed
        roi_area    = (x1-x0)*(y1-y0)
        step_size   = np.maximum(1,int(np.sqrt(roi_area)/20))

        #tck = interpolate.bisplrep(x, y, z, s=0)
        #znew = interpolate.bisplev(xnew, ynew, tck)
        roi3d       = self.imgXYZ[y0:y1:step_size,x0:x1:step_size,:]   
        if roi3d .shape[0] < 5:
            self.tprint('Too small region: %d' %step_size)
            step_size   = 1
            roi3d       = self.imgXYZ[y0:y1:step_size,x0:x1:step_size,:]   

        x,y,z       = roi3d[:,:,0].reshape((-1,1)), roi3d[:,:,1].reshape((-1,1)), roi3d[:,:,2].reshape((-1,1))

        # using svd to make the fit
        xyz1_matrix = np.hstack((x,y,z)) #,z*0+1))
        tvec        = xyz1_matrix[:,:3].mean(axis=0)
        xyz1_matrix = xyz1_matrix - tvec
        U, S, Vh    = np.linalg.svd(xyz1_matrix, full_matrices=True)
        ii          = np.argmin(S)
        vnorm       = Vh[ii,:]

        # keep orientation
        vnorm       = vnorm*np.sign(vnorm[2])

        # checking error
        err_std     = self.check_error(xyz1_matrix, vnorm)
        self.tprint('Fit error : %s' %str(err_std))

        # forming output
        #tvec        = xyz1_matrix[:,:3].mean(axis=0)
        #pose_norm   = np.hstack(tvec, vnorm.reshape((1,-1)))
        roi_params  = {'roi':roi, 'error': err_std, 'tvec': tvec, 'vnorm':vnorm }                               
        return roi_params
    
    def split_roi_recursively(self, roi, level = 0):
        # splits ROI on 4 regions and recursevly call 
        x0,y0,x1,y1     = roi
        #roi3d           = self.imgXYZ[y0:y1,x0:x1,:]   
        self.tprint('Processing level %d, region x = %d, y = %d' %(level,x0,y0))
        # check the current fit
        roi_params_f    = self.fit_plane(roi)
        roi_params_ret  = [roi_params_f]
        if roi_params_f['error'] < self.MIN_STD_ERROR:
            self.tprint('Fit is good enough x = %d, y = %d' %(x0,y0))
            return roi_params_ret

        # too small exit
        xs, ys          = int((x1 + x0)/2), int((y1 + y0)/2)
        if (xs - x0) < self.MIN_SPLIT_SIZE or (ys - y0) < self.MIN_SPLIT_SIZE:
            self.tprint('Min size is reached x = %d, y = %d' %(x0,y0))
            return roi_params_ret
        
        # 4 ROIs - accept the split if error of one of them is lower from the total
        roi_params_list = []
        roi_split   = [[x0,y0,xs,ys],[x0,ys,xs,y1],[xs,y0,x1,ys],[xs,ys,x1,y1]]
        for roi_s in roi_split:
            roi_params_prev = self.split_roi_recursively(roi_s, level + 1)
            # save locally
            #roi_params_list.append(roi_params_prev)
            roi_params_list = roi_params_list + roi_params_prev
            
        # extract each of the below and check the error
        makeTheSplit = False
        for roi_params_s in roi_params_list:
            #roi_params_s       = roi_params_prev[-1]
            # accept the split if twice lower (if noise of 4 split should be 2)
            if roi_params_s['error'] < roi_params_f['error']/2:
                makeTheSplit = True
                break

        # decide what to return
        if makeTheSplit:
            roi_params_ret = roi_params_list
            self.tprint('Split at level %d, region x = %d, y = %d' %(level,x0,y0))
        else:
            self.tprint('No split level %d, region x = %d, y = %d' %(level,x0,y0))

        return roi_params_ret
    
    def convert_roi_params_to_pose(self, roi_params):
        "converting params to the pose vector"
        tvec       = roi_params['tvec'].reshape((1,-1))
        vnorm      = roi_params['vnorm']  # 4x1 vector
        #rvec       = vnorm[:3].reshape((1,-1))
        #rvec       = rvec/np.linalg.norm(rvec)

        rvec       = vnorm.flatten() #reshape((-1,1))
        #rvec[3]    = 0 # kill DC
        rvec       = rvec/np.linalg.norm(rvec)

        #R           = Rot.from_quat(rvec).as_matrix()
        #R           = Rot.from_rotvec(rvec).as_matrix()
        #avec        = Rot.from_matrix(R).as_euler('zyx',degrees=True)
        #self.ang_vec= avec

        levl        = 0.1*tvec[0,2]
        pose_norm  = np.hstack((tvec, rvec.reshape((1,-1)),[[levl]]))
        #self.tprint('roi to pose')
        return pose_norm.flatten()

    def show_image_with_axis(self, img, poses = []):
        "draw results"
        axis_number = len(poses)
        if axis_number < 1:
            print('No poses found')
            
        # deal with black and white
        img_show = np.uint8(img) #.copy()
        if len(img.shape) < 3:
            img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
         
        for k in range(axis_number):
            
            avec    = poses[k][3:6] # orientation in degrees
            levl    = poses[k][6]   # level
            #R       = eulerAnglesToRotationMatrix(avec)
            R       = Rot.from_euler('zyx',avec, degrees = True).as_matrix()
            rvec, _ = cv.Rodrigues(R)
            tvec    = np.array(poses[k][:3], dtype = np.float32).reshape(rvec.shape) # center of the patch
            img_show= draw_axis(img_show, rvec, tvec, self.cam_matrix, self.cam_distort, len = levl)

        cv.imshow('Image & Axis', img_show)
        self.tprint('show done')
        ch = cv.waitKey()

    def show_image_with_rois(self, img, roi_params_ret = []):
        "draw results by projecting ROIs on image"

        axis_number = len(roi_params_ret)
        if axis_number < 1:
            print('No poses found')
            
        # deal with black and white
        img_show = np.uint8(img) #.copy()
        if len(img.shape) < 3:
            img_show = cv.applyColorMap(img_show, cv.COLORMAP_JET)
         
        for roi_p in roi_params_ret:

            pose    = self.convert_roi_params_to_pose(roi_p)            
            
            avec    = pose[3:6] # orientation in degrees
            levl    = pose[6]   # level
            #R       = eulerAnglesToRotationMatrix(avec)
            R       = Rot.from_euler('zyx',avec, degrees = True).as_matrix()
            rvec, _ = cv.Rodrigues(R)
            tvec    = np.array(pose[:3], dtype = np.float32).reshape(rvec.shape) # center of the patch
            img_show= draw_axis(img_show, rvec, tvec, self.cam_matrix, self.cam_distort, len = levl)

        cv.imshow('Image & Axis', img_show)
        self.tprint('show done')
        ch = cv.waitKey()

    def show_points_3d_with_normal(self, img3d, pose = None):
        "display in 3D"
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs,ys,zs       = img3d[:,:,0].reshape((-1,1)), img3d[:,:,1].reshape((-1,1)), img3d[:,:,2].reshape((-1,1))
        ax.scatter(xs, ys, zs, marker='.')
        
        if pose is not None:
            pose       = pose.flatten()
            # R          = Rot.from_euler('zyx',pose[3:6],degrees=True).as_matrix()
            # vnorm      = R[:,2]*pose[6]
            vnorm      = pose[3:6].flatten()*pose[6]
            xa, ya, za = [pose[0], pose[0]+vnorm[0]], [pose[1], pose[1]+vnorm[1]], [pose[2], pose[2]+vnorm[2]]
            ax.plot(xa, ya, za, 'r', label='Normal')


        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.show()

    def show_rois_3d_with_normals(self, roi_params_ret = [], roi_init = None):
        "display in 3D each ROI region with split"
        
        if len(roi_params_ret) < 1:
            self.tprint('roi_params_ret is empty')
            return

        # extract the initial ROI - to make the show more compact
        roi_init       = [0,0,self.frame_size[1], self.frame_size[0]] if roi_init is None else roi_init
        x0,y0,x1,y1    = roi_init

        if self.imgXYZ is None:
            self.tprint('Need init')
            return      

        img3d          = self.imgXYZ[y0:y1,x0:x1,:] 
        xs,ys,zs       = img3d[:,:,0].reshape((-1,1)), img3d[:,:,1].reshape((-1,1)), img3d[:,:,2].reshape((-1,1))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, marker='.')
        
        for roi_p in roi_params_ret:
            pose       = self.convert_roi_params_to_pose(roi_p)
            # R          = Rot.from_euler('zyx',pose[3:6],degrees=True).as_matrix()
            # vnorm      = R[:,2]*pose[6]
            vnorm      = pose[3:6].flatten()*pose[6]
            #self.tprint(str(vnorm))
            xa, ya, za = [pose[0], pose[0]+vnorm[0]], [pose[1], pose[1]+vnorm[1]], [pose[2], pose[2]+vnorm[2]]
            ax.plot(xa, ya, za, 'r', label='Normal')


        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.show() #block=False)        

    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt)

    def test_image(self):
        "test single image depth"
        img, roi = self.init_test_case(1)
        img_res  = self.fit_plane(img, roi)      


# ----------------------
#%% Tests
class TestPlaneMatcher(unittest.TestCase):
    def test_Convert(self):
        avec = np.random.randint(-180,180, size = (1,3)).flatten().astype(np.float32)
        R    = eulerAnglesToRotationMatrix(avec)
        bvec = rotationMatrixToEulerAngles(R)
        self.assertTrue(np.all(np.abs(avec - bvec) < 1e-6))

    def test_ImageShow(self):
        p = PlaneMatcher()
        p.init_image(1)
        poses = [[0,0,100,0,0,45,20]]
        p.show_image_with_axis(p.img,poses)
        self.assertFalse(p.img is None)

    def test_ChessPoseDetect(self):
        "understand pose ecomputations"
        p = PlaneMatcher()
        p.init_image(11)
        poses = p.detect_pose_in_chessboard()
        p.show_image_with_axis(p.img, poses)
        self.assertFalse(p.img is None)     

    def test_InitImg3d(self):
        "XYZ point cloud structure init"
        p = PlaneMatcher()
        p.init_image(1)
        img3d = p.init_img3d()
        self.assertFalse(img3d is None)    

    def test_ComputeImg3d(self):
        "XYZ point cloud structure init and compute"
        p       = PlaneMatcher()
        img     = p.init_image(1)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        self.assertFalse(imgXYZ is None)     

    def test_ShowImg3d(self):
        "XYZ point cloud structure init and compute"
        p       = PlaneMatcher()
        img     = p.init_image(1)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi     = p.init_roi(1)
        x0,y0,x1,y1 = roi
        roiXYZ    = imgXYZ[y0:y1,x0:x1,:]
        p.show_points_3d_with_normal(roiXYZ)
        self.assertFalse(imgXYZ is None)  
                     
    def test_FitPlane(self):
        "computes normal to the ROI"
        p       = PlaneMatcher()
        img     = p.init_image(4)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi     = p.init_roi(2)
        roip    = p.fit_plane(imgXYZ, roi)
        pose    = p.convert_roi_params_to_pose(roip)
        p.show_image_with_axis(p.img, pose)
                
        x0,y0,x1,y1 = roi
        roiXYZ       = imgXYZ[y0:y1,x0:x1,:]
        p.show_points_3d_with_normal(roiXYZ, pose)
        self.assertFalse(roip['error'] > 0.01)  

    def test_FitPlaneFail(self):
        "computes normal to the ROI but the image is bad at this location"
        p       = PlaneMatcher()
        img     = p.init_image(10)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi     = p.init_roi(1)
        roip    = p.fit_plane(imgXYZ, roi)
        pose    = p.convert_roi_params_to_pose(roip)
        p.show_image_with_axis(p.img, pose)
        self.assertTrue(roip['error'] > 0.01)          

    def test_FitPlaneDepthImage(self):
        "computes normal to the ROI"
        p       = PlaneMatcher()
        img     = p.init_image(13)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi     = p.init_roi(2)
        roip    = p.fit_plane(imgXYZ, roi)
        pose    = p.convert_roi_params_to_pose(roip)
        p.show_image_with_axis(p.img, pose)
                
        x0,y0,x1,y1 = roi
        roiXYZ       = imgXYZ[y0:y1,x0:x1,:]
        p.show_points_3d_with_normal(roiXYZ, pose)
        self.assertFalse(roip['error'] > 0.01)  

    def test_RoiSplit(self):
        "computes ROIS and splits if needed"
        p       = PlaneMatcher()
        p.MIN_STD_ERROR = 0.1
        img     = p.init_image(18)
        roi     = p.init_roi(3)
        img3d   = p.init_img3d(img)
        imgXYZ  = p.compute_img3d(img)
        roi_list= p.split_roi_recursively(roi)
        p.show_rois_3d_with_normals(roi_list, roi)
        p.show_image_with_rois(p.img, roi_list)

        for roi_s in roi_list:
            self.assertFalse(roi_s['error'] > 0.01)

# ----------------------
#%% App
class App:
    def __init__(self, src):
        self.cap   = RealSense()
        self.cap.change_mode('dep')

        self.frame = None
        self.paused = False
        self.tracker = PlaneMatcher()

        cv.namedWindow('plane')

    def run(self):
        while True:
            playing = not self.paused
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            vis = self.frame.copy()
            if playing:
                tracked = self.tracker.track(self.frame)
                for tr in tracked:
                    cv.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(tr.p1):
                        cv.circle(vis, (x, y), 2, (255, 255, 255))

            self.rect_sel.draw(vis)
            cv.imshow('plane', vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

if __name__ == '__main__':
    #print(__doc__)

    # import sys
    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0
    # App(video_src).run()

    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestPlaneMatcher("test_Convert"))
    #suite.addTest(TestPlaneMatcher("test_ImageShow"))
    #suite.addTest(TestPlaneMatcher("test_ChessPoseDetect")) # ok
    #suite.addTest(TestPlaneMatcher("test_InitImg3d")) # ok
    #suite.addTest(TestPlaneMatcher("test_ComputeImg3d")) # ok
    #suite.addTest(TestPlaneMatcher("test_ShowImg3d")) # 
    
    #suite.addTest(TestPlaneMatcher("test_FitPlane")) # ok
    #suite.addTest(TestPlaneMatcher("test_FitPlaneFail")) # 
    #suite.addTest(TestPlaneMatcher("test_FitPlaneDepthImage")) #

    suite.addTest(TestPlaneMatcher("test_RoiSplit")) 
   
    runner = unittest.TextTestRunner()
    runner.run(suite)
