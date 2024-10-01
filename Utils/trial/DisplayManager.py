"""
Display Manager - shows different 3D camera/object configurations

Environment:
    Zion  : dlcdev
    Avita : Levron

Usage : 
    from DisplayManager import DisplayManager
    c = DisplayManager()
    c.Test()
    
-----------------------------
 Ver    Date     Who    Descr
-----------------------------
0309    30.09.24 UD     Adopted
-----------------------------

use https://stackoverflow.com/questions/39408794/python-3d-pyramid for polygons
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
#from __future__ import print_function

import numpy as np
#import cv2 as cv
#import json
import os


import matplotlib.pyplot as plt
from matplotlib import cm
plt.style.use('dark_background')

# remove toolbar command
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None' 

# disable vefore compile
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

#from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
from scipy.spatial.transform import Rotation as scirot
from scipy.linalg import expm, inv


from numpy import linspace
try:
    from gui.ConfigManager import ConfigManager 
except:
    ConfigManager = None
    print('Running standalone mode')
#import pylab as m

import unittest
import logging
logger                  = logging.getLogger("robotai")

#%% Help functions
def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M   = np.identity(4)
#    M[1,1] = 0
#    M[1,2] = 1
#    M[2,1] = -1
#    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))
    
def listdir(file_extension,path='.'): 
   ''' return list of files with extension 'file_extension' in folder 'path'
       not case sensitive
       example: names = listdir( ['.jpg', '.jpeg'], 'C:\\db\\' ) 
   '''
   if isinstance(file_extension,str):
      file_extension=[file_extension] 
   b = []
   for j in range(np.size(file_extension)):
       a = [i for i in os.listdir(path) if (file_extension[j].lower() in i.lower())] 
       a = [i for i in a if (file_extension[j].lower()==i[-len(file_extension[j]):].lower())]
       a = [os.path.join(path,i) for i in a if (os.path.isfile(os.path.join(path,i)))]
       b = b + a  
   return b


#%% Help functins
def inversePerspective(rvec, tvec):
    #R, _ = cv.Rodrigues(rvec)
    R       = scirot.from_rotvec(rvec).as_matrix()
    R       = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    #invRvec, _ = cv.Rodrigues(R)
    invRvec  = scirot.from_matrix(R).as_rotvec()
    return invRvec, invTvec


def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def fromEulerToRvec(euler_angles_degrees):
    
    euler_angles_radians = euler_angles_degrees/180*np.pi
    
    rotation_mat         = eulerAnglesToRotationMatrix(euler_angles_radians.ravel())
    #rotation_vec, _      = cv.Rodrigues(rotation_mat)
    rotation_vec          = scirot.from_matrix(rotation_mat).as_rotvec()

    return rotation_vec

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# function that transforms to euler
def fromRvecToEuler(rvec) :

    #rotation_mat, _ = cv.Rodrigues(rvec)
    rotation_mat    = scirot.from_rotvec(rvec).as_matrix()
    euler_rad       = rotationMatrixToEulerAngles(rotation_mat)
    euler_angles_degrees = 180 * euler_rad/np.pi
    
    return euler_angles_degrees   


## test 
#angle_in        = np.array([100,-10.45, 90]).ravel()
#rvec            = fromEulerToRvec(angle_in)
#angle_out       = fromRvecToEuler(rvec)
#print('Error %s' %str(angle_in - angle_out))

#%% Spec matrices
def hat(v):
    return [[   0, -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]

def fromRvecTvecToHomogenious(r, t):
    res             = np.eye(4)
    res[0:3, 0:3]   = expm(hat(r))
    res[0:3, -1]    = t.reshape((3,))
    return res

def PoseToMatrix(pose):
    # pose = [Tx,Ty,Tz,Rx,Ry,Rz]
    rotSequence = 'xyz'
    pose        = pose.ravel()
    rvec        = (scirot.from_euler(rotSequence, pose[3:6], degrees=True).as_rotvec()).reshape((3,1))        
    #R, _        = cv.Rodrigues(rvec)
    R           = scirot.from_rotvec(rvec).as_matrix()
    Mo          = np.eye(4,4)
    Mo[0:3,0:3] = R
    Mo[0:3,3]   = pose[0:3]    
    return Mo

def MatrixToPose(Mo):
    # pose = [Tx,Ty,Tz,Rx,Ry,Rz]
    rotSequence = 'xyz'
    pose        = np.zeros((6,1)).reshape((6,1))
    
    R           = Mo[0:3,0:3]
    #rvec, _     = cv.Rodrigues(R)
    rvec        = scirot.from_matrix(R).as_rotvec()
    pose[3:6]   = (scirot.from_rotvec(rvec.ravel()).as_euler(rotSequence,degrees=True)).reshape((3,1)) 
    pose[0:3]   = Mo[0:3,3].reshape((3,1))
    
    return pose.ravel()

## test 
#pose_in         = np.array([60, 91.5,0.555,100,-0.45,  90]).ravel()
#M               = PoseToMatrix(pose_in)
#pose_out        = MatrixToPose(M)
#print('Error %s' %str(pose_in - pose_out))
#
#rvec,_          = cv.Rodrigues(M[0:3,0:3])
#tvec            = M[0:3, -1]
#M2              = fromRvecTvecToHomogenious(rvec, tvec)
#print('Homog Error %s' %str(M - M2))
#%% =========================

def fromPoseToHomogeneousMatrix(pose):
    
    pose        = pose.ravel()
    rvec        = fromEulerToRvec(pose[3:6]) 
    #rvec        = extrinsics[idx,3:6]
    #R, _       = cv.Rodrigues(rvec)
    R           = scirot.from_rotvec(rvec).as_matrix()
    Mo          = np.eye(4,4)
    Mo[0:3,0:3] = R
    Mo[0:3,3]   = pose[0:3]
    return Mo
    
def fromHomogeneousMatrixToPose(Mo):
    # converts 4x4 mat represenation to tvex and rvec
    pose          = np.zeros((1,6))
    X           = Mo
    #rvec, _     = cv.Rodrigues(X[0:3,0:3])
    rvec        = scirot.from_matrix(X[0:3,0:3]).as_rotvec()
    evec        = fromRvecToEuler(rvec).reshape((1,3))            
    tvec        = X[0:3,3].reshape((1,3))
    
    # assign
    pose        = np.hstack((tvec,evec))            
    return pose

def applyPoseTransform(pose0,poseT):
    
    M0         = fromPoseToHomogeneousMatrix(pose0)
    MT         = fromPoseToHomogeneousMatrix(poseT)
    M1         = MT.dot(M0)
    pose1      = fromHomogeneousMatrixToPose(M1)
    return pose1


pose_in     = np.array([10,-9.3, 1, 89, -60, 120])
M           = fromPoseToHomogeneousMatrix(pose_in)
pose_out    = fromHomogeneousMatrixToPose(M)
print('Error %s' %str(pose_in - pose_out))

pose_tr     = np.array([-10,9.3, 0, 0, 0, 0])
pose_out    = applyPoseTransform(pose_in, pose_tr)
print(pose_out)

#%%---------------------------------------------------------------------------------------------------------------------------------------
# 3d Axis frame
class Frame:
    def __init__(self, fname = 'frame'):
        self.name               = fname  # H,W,Depth
        self.extrisincs         = np.zeros((0,6))
        self.params_model       = {'width':10, 'height':10,'depth':10}
        self.x_frame            = self.create_frame_model(20)
        self.h_frame            = [None]*len(self.x_frame) # assocoation to graphocs   
        self.h_text             = None
        self.draw_axis          = True
        
    def create_frame_model(self, height = 10):
        # create frame axis
        
        # draw board frame axis
        X_frame1        = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [height, 0, 0]
    
        X_frame2        = np.ones((4,2))
        X_frame2[0:3,0] = [0, 0, 0]
        X_frame2[0:3,1] = [0, height, 0]
    
        X_frame3        = np.ones((4,2))
        X_frame3[0:3,0] = [0, 0, 0]
        X_frame3[0:3,1] = [0, 0, height]
        
        return [X_frame1, X_frame2, X_frame3]        

# Sub class - Hangers
class Hanger(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'hanger'  # H,W,Depth
        self.params_model       = {'width':30, 'height':20,'depth':10}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        shift          = 5
        
        outv           = []
        # draw point as a square
        X_point         = np.ones((4,13))
        X_point[0:3,0] = [0,0,-depth*4]
        X_point[0:3,1] = [0,0,0]
        X_point[0:3,2] = [0,-height/2,depth]
        X_point[0:3,3] = [0,-height/2,depth*2]
        X_point[0:3,4] = [0,0,depth*3]
        X_point[0:3,5] = [0,height/2,depth*1.5]
        X_point[0:3,6] = [0,height/2,depth*1.5-shift]
        X_point[0:3,7] = [0,0,depth*3-shift]
        X_point[0:3,8] = [0,-height/2,depth*2-shift]
        X_point[0:3,9] = [0,-height/2,depth-shift]  
        X_point[0:3,10]= [0,0,0-shift]  
        X_point[0:3,11]= [0,0,-depth-shift]  
        X_point[0:3,12]= [0,0,-depth*4]  
        
        # output should be a list
        outv.append(X_point)
            
        return outv 

    
# Sub class - Hangers
class Separator(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'separator'  # H,W,Depth
        self.params_model       = {'width':10, 'height':10,'depth':10}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        shift          = 5

        
        outv           = []
        # draw point as a square
        X_point         = np.ones((4,15))
        X_point[0:3,0] = [0, height*3,-depth]
        X_point[0:3,1] = [0, 0,       0]
        X_point[0:3,2] = [0, height/2,depth]
        X_point[0:3,3] = [0, height/2,depth*2]
        X_point[0:3,4] = [0, 0,       depth*3]
        X_point[0:3,5] = [0, -height/2,depth*1.5]
        X_point[0:3,6] = [0, -height/2,depth*1.5-shift]
        X_point[0:3,7] = [0, 0,       depth*3-shift]
        X_point[0:3,8] = [0, height/2,depth*2-shift]
        X_point[0:3,9] = [0, height/2,depth-shift]  
        X_point[0:3,10]= [0, 0+shift, 0]  
        X_point[0:3,11]= [0, -height*3,-depth]  
        X_point[0:3,12]= [0,-height*3,-depth-shift]  
        X_point[0:3,13]= [0, height*3,-depth-shift]  
        X_point[0:3,14]= [0, height*3,-depth]         
        
        # output should be a list
        outv.append(X_point)
 
        return outv     
    
# Sub class 
class Gripper(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'gripper'  # H,W,Depth
        self.params_model       = {'width':40, 'height':80,'depth':40}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):
        
        f_scale = self.params_model['depth']
        width   = self.params_model['width']/2
        height  = self.params_model['height']/2
            
        # draw finder plane
        X_upr_plane        = np.ones((4,5))
        X_upr_plane[0:3,0] = [-width, height, f_scale]
        X_upr_plane[0:3,1] = [width, height,  f_scale]
        X_upr_plane[0:3,2] = [width,      0,  0]
        X_upr_plane[0:3,3] = [-width,     0,  0]
        X_upr_plane[0:3,4] = [-width, height, f_scale]
    
        # draw triangle above the finder plane
        X_triangle = np.ones((4,3))
        X_triangle[0:3,0] = [-width, -height, f_scale]
        X_triangle[0:3,1] = [0, -height/2, f_scale * 1.1]
        X_triangle[0:3,2] = [width, -height, f_scale]
        X_finger_upr      = X_triangle
    
        # draw lower finder plane
        X_low_plane        = np.ones((4,5))
        X_low_plane[0:3,0] = [-width, -height, f_scale]
        X_low_plane[0:3,1] = [width,  -height, f_scale]
        X_low_plane[0:3,2] = [width,      0,  0]
        X_low_plane[0:3,3] = [-width,     0,  0]
        X_low_plane[0:3,4] = [-width, -height, f_scale]
        
        # draw triangle above the finder plane
        X_triangle = np.ones((4,3))
        X_triangle[0:3,0] = [-width, height, f_scale]
        X_triangle[0:3,1] = [0, height/2, f_scale * 1.1]
        X_triangle[0:3,2] = [width, height, f_scale]
        X_finger_low      = X_triangle
            
        # output should be a list
        outv            = [X_upr_plane, X_finger_upr, X_low_plane, X_finger_low]
        
        # move zeros zero point to the tip of the gripper  
        for k in range(len(outv)):
            outv[k][2,:]       = outv[k][2,:] - f_scale          

        return outv

               
class Camera(Frame):
    def __init__(self, size = (40,60,60)):
        super().__init__()
        self.name               = 'camera'  # H,W,Depth
        self.params_model       = {'width':size[0], 'height':size[1],'depth':size[2]}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):
        
        f_scale             = self.params_model['depth']
        width               = self.params_model['width']/2
        height              = self.params_model['height']/2
        
        # draw image plane
        X_img_plane = np.ones((4,5))
        X_img_plane[0:3,0] = [-width, height, f_scale]
        X_img_plane[0:3,1] = [width, height, f_scale]
        X_img_plane[0:3,2] = [width, -height, f_scale]
        X_img_plane[0:3,3] = [-width, -height, f_scale]
        X_img_plane[0:3,4] = [-width, height, f_scale]
    
        # draw triangle above the image plane
        X_triangle = np.ones((4,3))
        X_triangle[0:3,0] = [-width, height, f_scale]
        X_triangle[0:3,1] = [0, 2*height, f_scale]
        X_triangle[0:3,2] = [width, height, f_scale]
    
        # draw camera
        X_center1 = np.ones((4,2))
        X_center1[0:3,0] = [0, 0, 0]
        X_center1[0:3,1] = [-width, height, f_scale]
    
        X_center2 = np.ones((4,2))
        X_center2[0:3,0] = [0, 0, 0]
        X_center2[0:3,1] = [width, height, f_scale]
    
        X_center3 = np.ones((4,2))
        X_center3[0:3,0] = [0, 0, 0]
        X_center3[0:3,1] = [width, -height, f_scale]
    
        X_center4 = np.ones((4,2))
        X_center4[0:3,0] = [0, 0, 0]
        X_center4[0:3,1] = [-width, -height, f_scale]
    
        # output should be a list
        #outv            = [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]
        outv            = [X_img_plane, X_center1, X_center2, X_center3, X_center4]
         # draw board with axis

        return outv
       
class Board(Frame):
    def __init__(self, size = (500,500)):
        super().__init__()
        self.name               = 'board'  # H,W,Depth
        self.params_model       = {'width':size[0], 'height':size[1],'depth':1}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):   
        # show board

        #f_scale = params_model['depth']
        width         = self.params_model['width']
        height        = self.params_model['height']
    
        # draw calibration board
        X_board         = np.ones((4,5))
        #X_board_cam = np.ones((extrinsics.shape[0],4,5))
        X_board[0:3,0] = [0,0,0]
        X_board[0:3,1] = [width,0,0]
        X_board[0:3,2] = [width,height,0]
        X_board[0:3,3] = [0,height,0]
        X_board[0:3,4] = [0,0,0]
    
        # output should be a list
        outv            = [X_board]
         # draw board with axis
        return outv
    
# 3D box
class Box3D(Frame):
    def __init__(self, size = (200,500,300)):
        super().__init__()
        self.name               = 'board'  # H,W,Depth
        self.params_model       = {'width':size[0], 'height':size[1],'depth':size[2]}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):   
        # show board

        #f_scale = params_model['depth']
        dx        = self.params_model['width']
        dy        = self.params_model['height']
        dz        = self.params_model['depth']
    
        # draw calibration board
        X_board         = np.ones((4,14))
        #X_board_cam = np.ones((extrinsics.shape[0],4,5))
        X_board[0:3,0] = [0,0,0]
        X_board[0:3,1] = [dx,0,0]
        X_board[0:3,2] = [dx,dy,0]
        X_board[0:3,3] = [0,dy,0]
        X_board[0:3,4] = [0,0,0]
        X_board[0:3,5] = [0,0,dz]
        X_board[0:3,6] = [dx,0,dz]
        X_board[0:3,7] = [dx,dy,dz]
        X_board[0:3,8] = [0,dy,dz]
        X_board[0:3,9] = [0,0,dz]
        X_board[0:3,10] = [dx,0,dz]
        X_board[0:3,11] = [dx,0,0]
        X_board[0:3,12] = [0,0,0]
        X_board[0:3,13] = [0,0,dz]
        # output should be a list
        outv          = [X_board]
         # draw board with axis
        return outv    
    
# polygon
class Base(Frame):
    def __init__(self, radius = 500):
        super().__init__()
        self.name               = 'base'  # H,W,Depth
        self.params_model       = {'width':radius, 'height':100,'depth':100,'side_number':16}
        
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs    
        
    def create_model(self):   
        # show board

        #f_scale = params_model['depth']
        radius          = self.params_model['width']  # 
        height          = self.params_model['height']
        side_num        = self.params_model['side_number']
        
        t_data          = np.linspace(0, 2*np.pi, side_num)
        y_data          = np.sin(t_data)*radius + radius
        x_data          = np.cos(t_data)*radius + radius        
    
        # draw calibration board
        X_board         = np.ones((4,side_num*2))
        #X_board_cam = np.ones((extrinsics.shape[0],4,5))
        X_board[0,:]    = np.hstack((x_data,x_data))
        X_board[1,:]    = np.hstack((y_data,y_data))
        X_board[2,:]    = np.hstack((x_data*0,x_data*0+height))

    
        # output should be a list
        outv            = [X_board]
         # draw board with axis
        return outv 
    
# Sub class - Tool
class Tool(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'tool'  # H,W,Depth
        self.params_model       = {'width':10, 'height':10,'depth':300}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):

        width          = self.params_model['width']
        height         = self.params_model['height']
        length         = self.params_model['depth']
        shift          = 5
        
        # createing 3D box
        a               = np.meshgrid([0,1],[0,1],[0,1])
        bbox           = np.array(a).T.reshape(-1,3).T
        
        outv           = []
        # draw point as a square
        X_point         = np.ones((4,bbox.shape[1]))
        X_point[0,:]    = bbox[0,:]*width
        X_point[1,:]    = bbox[1,:]*height
        X_point[2,:]    = bbox[2,:]*length
               
        # output should be a list
        outv.append(X_point)
 
        return outv     
        
class Line(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'line'  # H,W,Depth
        self.params_model       = {'width':0, 'height':800,'depth':5}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):  
        # show line

        height          = self.params_model['height']
        X_frame1        = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [0, 0, height]

        # output should be a list
        outv            = [X_frame1]
        return outv  
    
class Ray(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'ray'  # H,W,Depth
        self.params_model       = {'width':10, 'height':10,'depth':10}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):   

        # show multiple rays
        # width, height, depth must be arrays of the same size

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        
        outv           = []
        for k in range(len(width)):
            X_frame1        = np.ones((4,2))
            X_frame1[0:3,0] = [0, 0, 0]
            X_frame1[0:3,1] = [width[k], height[k], depth[k]]
    
            # output should be a list
            outv.append(X_frame1)
            
        return outv    
    
class Point(Frame):
    def __init__(self):
        super().__init__()
        self.name               = 'point'  # H,W,Depth
        self.params_model       = {'width':5, 'height':5,'depth':5}
        self.x_model            = self.create_model()
        self.h_model            = [None]*len(self.x_model) # assocoation to graphocs         
        
    def create_model(self):    

        # show multiple points
        # width, height, depth must be arrays of the same size

        width          = self.params_model['width']
        height         = self.params_model['height']
        depth          = self.params_model['depth']
        
        outv           = []
        # draw point as a square
        X_point         = np.ones((4,10))
        X_point[0:3,0] = [0,0,0]
        X_point[0:3,1] = [width,0,0]
        X_point[0:3,2] = [0,height,0]
        X_point[0:3,3] = [0,0,0]
        X_point[0:3,4] = [0,0,depth]
        X_point[0:3,5] = [0,height,0]
        X_point[0:3,6] = [0,0,0]
        X_point[0:3,7] = [0,0,depth]
        X_point[0:3,8] = [width,0,0]
        X_point[0:3,9] = [0,0,0]  
        
        # output should be a list
        outv.append(X_point)
            
        return outv   
    
    #=========================================
    def create_object_model(self, params_model, draw_frame_axis=True):
    #def create_object_model(self, faces, labels, draw_frame_axis=True):
        """ Faces is a list of lists, each representing one face. The inner list is made of labels
        Labels is a dictionary, where each key is a string of a label, and its value is its coordinate
        square_size is from config file
        """
        
        faces       = params_model['faces']
        labels      = params_model['labels']
        
        # longest = max(faces, key=lambda x:len(x))
        faces_np = [] #list of frames 
        for idx in range(len(faces)): # for each frame
            face_size = len(faces[idx])
            face = np.ones((4, face_size+1)) #make an array of 4x7(or the points in the face+1)
            faces_coords = [labels[pt] for pt in faces[idx]] #get number coordinate values for each of the faces
            for idy in range(len(faces_coords)): #for each coordinate
                # size = len(faces_coords[idy])
                face[0:3, idy] = faces_coords[idy] #add the point into the array
            face[0:3, idy+1] = faces_coords[0] #add the first coord as the last point in the array
            faces_np.append(face)
            
     
        if draw_frame_axis:
            height      = 20
            X_frame     = self.create_frame_model(height)
            faces_np.extend(X_frame)
            
        return faces_np

#%%---------------------------------------------------------------------------------------------------------------------------------------
# Main class
class DisplayManager:
    def __init__(self, config = None):
        self.cfg                = config
        self.debugOn            = False
        #self.path               = os.getcwd()             # path to the camera folder
        
        # configuration file params
        self.camMatrix         = []
        self.camDist           = []
        self.modelPoints       = []
        
        # camera gripper matrix and pose
        self.poseGC             = np.zeros((1,6))

        # gripper to base multiple poses
        self.poseBG             = np.zeros((0,6))      # could be Nx6 array
        
        # camera to base multiple poses
        self.poseBC             = np.zeros((0,6))      
        
        # boards to base multiple poses
        self.poseBB             = np.zeros((0,6))          
        
        # gui handles dictionary for real time update
        self.gui_handles       = {}
        
        # GUI assist
        self.path               = ''
        
#        # should not be done - connect to external
#        if self.cfg is None:
#            self.cfg = ConfigManager()    
        
        # graphics
        self.fig                = None
        self.ax                 = None
        self.min_values         = np.array([0,-510,-10]) # min axis range
        self.max_values         = np.array([1000,510,710]) # max axis range
        
        # object list to remmenber
        self.object_list_current = []
        
        self.Init()
        self.Print("Display Manager Created")
        
    def Init(self):
        
        self.min_values         = np.array([0,-510,-10]) # min axis range
        self.max_values         = np.array([1000,510,710]) # max axis range
        
        self.object_list_current = []
        
        self.GetParamsFromConfigFile()
        
        return True
        
    def GetParamsFromConfigFile(self):
        "update all parameters from config file"
 
        if self.cfg is None:
            self.Print('Configuration is not intialized')
            return    
        
        if not self.cfg.IsFileExist():
            self.Print('Configuration is not intialized properly')
            return    
        
        cfg                 = self.cfg.ReadPlainconfig()
                
        # Update
        cmtrx               = cfg['CamMatrix']
        cdist               = cfg['CamDist']
        mpoints             = cfg['model_points']    
                
        self.camMatrix         = np.array(cmtrx).astype(np.float32)
        self.camDist           = np.array(cdist).reshape((-1,1)).astype(np.float32)
        self.modelPoints       = np.array(mpoints).astype(np.float32)
        
        rvecGC, tvecGC      = self.cfg.GetCameraGripperTransform()
        self.poseGC         = self.cfg.TransformRvecTvecToPose(rvecGC, tvecGC,'xyz')
        
       
        self.Print('Params configuration is read')
        return 
    
    def SetGripperPose(self, poseGB = None):
        # set gripper positions
        if poseGB is None:
            self.Print('Gripper Base pose is not specified and is not updated','W')
            return         
                    
        self.poseGB = poseGB
        return
    
    def CheckGripperPose(self):
        # check if gripper positions are ok
        ret = False
        if self.poseGB is None:
            self.Print('Gripper Base pose is not specified and is not updated','W')
            return ret
            
        if self.poseGB.shape[1]!=6:
            self.Print('Gripper Base pose must have 6 parameters','W')
            return ret  

        if self.poseGB.shape[0] < 1:
            self.Print('Gripper Base pose must have at least one pose','W')
            return ret         
                    
        return True

    
    def GetGripperParams(self, extrinsics_grip = None):
        # configures gripper positions
        object_grip = []
        
        # update if provided
        self.SetGripperPose(extrinsics_grip)
        
        ret = self.CheckGripperPose()
        if ret is False:
            return object_grip

        # extrinsics to objects    
        for k in range(len(extrinsics_grip)):
            object_cam                = Gripper()
            object_cam.extrinsics     = extrinsics_grip[k] 
            object_grip.append(object_cam)           

        return object_grip
    
    def GetCameraParams(self, extrinsics_cam = None, sizeBB = (40,60,60)):
        # configures gripper positions
        params_cam          = []
       
        if extrinsics_cam is None:
        
            ret = self.CheckGripperPose()
            if ret is False:
                return params_cam
            
            # init important  params
            mtrxGC              = PoseToMatrix(self.poseGC)
            extrinsics_grip     = self.poseGB
            
            # convert camera params   
            extrinsics_cam      = np.zeros((0,6))
            for k in range(len(extrinsics_grip)):
                poseBG          = extrinsics_grip[k]
                mtrxBG          = PoseToMatrix(poseBG)
                mtrxBC          = np.dot(mtrxBG, mtrxGC)
                poseBC          = MatrixToPose(mtrxBC)
                extrinsics_cam  = np.vstack((extrinsics_cam,poseBC))
            
        # extrinsics to objects    
        for k in range(len(extrinsics_cam)):
            object_cam                = Camera(sizeBB)
            object_cam.extrinsics     = extrinsics_cam[k] 
            params_cam.append(object_cam)            

        return params_cam
    
    def GetBoardParams(self, poseBB = None, sizeBB = (200,200)):
        # configures gripper positions
        params_board = []
        
        # update if provided
        if poseBB is None:
            self.Print('Gripper Base pose is not specified and is not updated','W')
            return params_board
            
        if poseBB.shape[1]!=6:
            self.Print('Gripper Base pose must have 6 parameters','W')
            return params_board  

        if poseBB.shape[0] < 1:
            self.Print('Gripper Base pose must have at least one pose','W')
            return params_board 
        
        self.poseBB         = poseBB
        
        # import argparse
        extrinsics_cam      = np.zeros((0,6))
        for k in range(len(poseBB)):
            extrinsics_cam  = np.vstack((extrinsics_cam,poseBB[k]))
            
            object_board                = Board(sizeBB)
            object_board.extrinsics     = poseBB[k] 
            params_board.append(object_board)
     
        # board just for reference
        return params_board
    
    def GetBaseParams(self, poseBB = None, baseRadius = None):
        # configures gripper positions
        params_base = []
        
        # update if provided
        if poseBB is None:
            self.Print('Base pose is not specified and is not updated','W')
            return params_base
            
        if poseBB.shape[1]!=6:
            self.Print('Base pose must have 6 parameters','W')
            return params_base  

        if poseBB.shape[0] < 1:
            self.Print('Base pose must have at least one pose','W')
            return params_base 
        
        if baseRadius is None:
            baseRadius = [500]*len(poseBB)
            
        if len(baseRadius) != len(poseBB):
            self.Print('baseRadius must be equal to number of poses','W')
            return params_base
        
        self.poseBB         = poseBB
        
        # import argparse
        extrinsics_cam      = np.zeros((0,6))
        for k in range(len(poseBB)):
            extrinsics_cam              = np.vstack((extrinsics_cam,poseBB[k]))
            
            object_board                = Base(baseRadius[k])
            object_board.extrinsics     = poseBB[k] 
            params_base.append(object_board)
     
        # board just for reference
        return params_base    
    
    
    def GetRayParams(self, rayPixels):
        # computes rays in camera coordinate system using pixel coordinates
        # rayPixels is a List of K of 2 x N array of K pixels when N is number of cameras 
        params_ray = []
        
        # checks
        if self.camMatrix.shape[0] != 3:
            self.Print('Camera Intrinsic matrix is not initialized','W')
            return params_ray   

        # checks
        if self.poseBC.shape[0] != len(rayPixels):
            self.Print('Number of camera poses does not match number of pixel pairs ','W')
            return params_ray   
     
        if rayPixels[0].shape[1] != 2:
            self.Print('Pixels rays must be 2D ','W')
            return params_ray         

        extrinsics_cam      = self.poseBC          
        
        # convert rays
        sensorScale         = 300     
        camMtrxInv          = inv(self.camMatrix)  
        for k in range(self.poseBC.shape[0]):
            
            ray_pixels_camera       = rayPixels[k].T # need to be 2 x N
            ray_pixels_homog        = np.vstack((ray_pixels_camera, np.ones((1,ray_pixels_camera.shape[1]))))
            rayCoords               = np.dot(camMtrxInv, ray_pixels_homog)*sensorScale
     
            obj_width, obj_height, obj_depth   = rayCoords[0,:], rayCoords[1,:], rayCoords[2,:]   
            
            object_ray              = Ray()
            object_ray.extrinsic    = extrinsics_cam[k,:].reshape((1,6))
            object_ray.param_model  = {'width':obj_width,'height':obj_height, 'depth':obj_depth} 
            params_ray.append(object_ray)

        return params_ray
    
    def GetPointParams(self, points3D):
        # define params for the points in 3D
        
        params_points = []
        
        if points3D.shape[1] != 3:
            self.Print('Points must be 3D ','W')
            return params_points    
        
        pointNum = points3D.shape[0] 
        for k in range(pointNum):

            extrinsics_point            = np.hstack(([points3D[k,:], points3D[k,:]*0]))                
            object_point                = Point()
            object_point.extrinsics     = extrinsics_point 
            params_points.append(object_point)

        return params_points   
       
    
    def GetLineParams(self, poseLines):
        # define params for the lines in 3D
        
        params_lines = []
        
        if poseLines is None:
            self.Print('Line Base pose is not specified and is not updated','W')
            return params_lines
            
        if poseLines.shape[1]!=6:
            self.Print('Line Base pose must have 6 parameters','W')
            return params_lines  

        if poseLines.shape[0] < 1:
            self.Print('Line Base pose must have at least one pose','W')
            return params_lines 
        
        
        # import argparse
        for k in range(len(poseLines)):
            
            object_line                = Line()
            object_line.extrinsics     = poseLines[k] 
            params_lines.append(object_line)
 

        return params_lines      
    
    def GetHangerParams(self, extrinsics_hang):
        # define params for the hangerss
        
        params_hangers = []
        
        if extrinsics_hang is None:
            self.Print('Hanger Base pose is not specified and is not updated','W')
            return params_hangers
            
        if extrinsics_hang.shape[1]!=6:
            self.Print('Hanger Base pose must have 6 parameters','W')
            return params_hangers  

        if extrinsics_hang.shape[0] < 1:
            self.Print('Hanger Base pose must have at least one pose','W')
            return params_hangers 
        
        # import argparse
        for k in range(len(extrinsics_hang)):
            
            object_line                = Hanger()
            object_line.extrinsics     = extrinsics_hang[k] 
            params_hangers.append(object_line)
 
        return params_hangers  
    
    def GetSeparatorParams(self, extrinsics_separator):
        # define params for the hangerss
        
        params_separator = []
        
        if extrinsics_separator is None:
            self.Print('Separator Base pose is not specified and is not updated','W')
            return params_separator
            
        if extrinsics_separator.shape[1]!=6:
            self.Print('Separator Base pose must have 6 parameters','W')
            return params_separator  

        if extrinsics_separator.shape[0] < 1:
            self.Print('Separator Base pose must have at least one pose','W')
            return params_separator 
        
        # import argparse
        for k in range(len(extrinsics_separator)):
            
            object_line                = Separator()
            object_line.extrinsics     = extrinsics_separator[k] 
            params_separator.append(object_line)
 
        return params_separator 
    
    def GetToolParams(self, extrinsics_par):
        # define params for the hangerss
        
        params_t = []
        
        if extrinsics_par is None:
            self.Print('Separator Base pose is not specified and is not updated','W')
            return params_t
            
        if extrinsics_par.shape[1]!=6:
            self.Print('Separator Base pose must have 6 parameters','W')
            return params_t  

        if extrinsics_par.shape[0] < 1:
            self.Print('Separator Base pose must have at least one pose','W')
            return params_t 
        
        # import argparse
        for k in range(len(extrinsics_par)):
            
            object_line                = Tool()
            object_line.extrinsics     = extrinsics_par[k] 
            params_t.append(object_line)
 
        return params_t    
    
    def GetBoxParams(self, poseBB = None, sizeBB = [100,50,200]):
        # configures box positions and params
        params_base = []
        
        # update if provided
        if poseBB is None:
            self.Print('Base pose is not specified and is not updated','W')
            return params_base
            
        if poseBB.shape[1]!=6:
            self.Print('Base pose must have 6 parameters','W')
            return params_base  

        if poseBB.shape[0] < 1:
            self.Print('Base pose must have at least one pose','W')
            return params_base 
        
        self.poseBB         = poseBB
        
        # import argparse
        extrinsics_cam      = np.zeros((0,6))
        for k in range(len(poseBB)):
            extrinsics_cam              = np.vstack((extrinsics_cam, poseBB[k]))
            
            object_board                = Box3D(sizeBB)
            object_board.extrinsics     = poseBB[k] 
            params_base.append(object_board)
     
        # board just for reference
        return params_base      

    def ClearScene(self):
        # sets view range of the scene
        if self.ax is None:
            return
        self.ax.cla()
        self.Init() # default xmin-xmax
        self.ScaleScene(self.ax)
        

    def ScaleScene(self, ax):
        # sets view range of the scene
        if ax is None:
            return
        
        X_min = self.min_values[0]
        X_max = self.max_values[0]
        Y_min = self.min_values[1]
        Y_max = self.max_values[1]
        Z_min = self.min_values[2]
        Z_max = self.max_values[2]

        range_x = np.array([X_max-X_min]).max() * 1.2 # to get some volume 2.0
        range_y = np.array([Y_max-Y_min]).max() * 1.2 
        range_z = np.array([Z_max-Z_min]).max() * 1.2
        max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 1.9 # to get some volume 2.0
        range_x = range_y = range_z = max_range
    
        mid_x = (X_max+X_min) * 0.5
        mid_y = (Y_max+Y_min) * 0.5
        mid_z = (Z_max+Z_min) * 0.5
        ax.set_xlim(mid_x - range_x, mid_x + range_x)
        ax.set_ylim(mid_y - range_y, mid_y + range_y)
        ax.set_zlim(mid_z - range_z, mid_z + range_z)  
        
        #ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
        ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
        #set_axes_equal(ax) # IMPORTANT - this is also required
        #ax.axis('equal')
        

    def InitScene(self, ax = None, fig = None):
        # init 3D scene
        if ax is None or fig is None:
            fig = plt.figure(1)
            plt.clf() 
            plt.ion() 
            #fig.canvas.set_window_title('3D Scene')
            ax = fig.add_subplot(projection='3d')
            fig.tight_layout()
            
            #ax.set_proj_type('ortho')
            
            #self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            #self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            #self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        #ax.set_aspect("equal")
        self.ScaleScene(ax)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
        
        #ax.set_title('Object Visualization')
        plt.show()
        self.Print('Scene rendering is done')
        
        self.fig = fig
        self.ax = ax
        return ax  
    
    def ComputeTransformMatrix(self, extrinsics, patternCentric):
                    
        rvec        = fromEulerToRvec(extrinsics[3:6]) 
        #R, _        = cv.Rodrigues(rvec)
        R           = scirot.from_rotvec(rvec).as_matrix()
        cMo         = np.eye(4,4)
        cMo[0:3,0:3] = R
        cMo[0:3,3]   = extrinsics[0:3]
            
        if patternCentric:
            cMo     = inverse_homogeneoux_matrix(cMo)
            
        return cMo

#            ext = np.concatenate((tvec, rvec), axis=0).reshape(1,6)
#            if extrinsics_camera is None:
#                extrinsics_camera = ext
#            else:
#                extrinsics_camera = np.vstack((extrinsics_camera,ext))
#                
#            return extrinsics_camera

    def DrawObjects(self, ax, object_list, patternCentric = False):
        # check
        
        min_val     = np.inf
        max_val     = -np.inf
        min_values  = np.tile(min_val,3)
        max_values  = np.tile(max_val,3)
        
        object_num  = len(object_list)
        if object_num < 1:
            return min_values, max_values
    
        cm_subsection   = linspace(0.0, 1.0, object_num)
        #colors          = [ cm.rainbow(x) for x in cm_subsection ]
        colors          = [ cm.Pastel1(x) for x in cm_subsection ]
        #rotSequence     = 'xyz' Paired

        for k in range(object_num):
            
            # get params
            extrinsics  = object_list[k].extrinsics
            x_moving    = object_list[k].x_model
            x_frame     = object_list[k].x_frame
            draw_axis   = object_list[k].draw_axis
            name_model  = object_list[k].name
            
            # compute transform
            cMo         = self.ComputeTransformMatrix(extrinsics, patternCentric)

            # colors    
            clr          = colors[k]
            clr         = [0.5, 0.5, 0.5] if name_model == 'ray' else clr     
            
            oneTime      = name_model != 'ray'
            part_num     = len(x_moving)
            for i in range(part_num):
                # transform
                X       = cMo.dot(x_moving[i])
                
                # draw
                if object_list[k].h_model[i] is None:
                    object_list[k].h_model[i],  = ax.plot3D(X[0,:], X[1,:], X[2,:], color=clr)
                else:
                    object_list[k].h_model[i].set_data(X[0,:], X[1,:])
                    object_list[k].h_model[i].set_3d_properties(X[2,:])
                    
                # show text
                if oneTime:
                    oneTime = False
                    if object_list[k].h_text is None:
                        object_list[k].h_text = ax.text(X[0,0], X[1,0]-10, X[2,0]+20, str(k), color=colors[k])
                    else:
                        object_list[k].h_text.set_x(X[0,0])
                        object_list[k].h_text.set_y( X[1,0]-10)
                        object_list[k].h_text.set_3d_properties(X[2,0]+20)
                    
                    
                min_values = np.minimum(min_values, X[0:3,:].min(1))
                max_values = np.maximum(max_values, X[0:3,:].max(1))
                
            if draw_axis is False:
                continue
                        
            part_num     = len(x_frame)
            clr          = 'rgb'
            for i in range(part_num):
                # transform
                X       = cMo.dot(x_frame[i])
                
                if object_list[k].h_frame[i] is None:
                    object_list[k].h_frame[i],  = ax.plot3D(X[0,:], X[1,:], X[2,:], color=clr[i])
                else:
                    object_list[k].h_frame[i].set_data(X[0,:], X[1,:])  
                    object_list[k].h_frame[i].set_3d_properties(X[2,:])
                
                min_values = np.minimum(min_values, X[0:3,:].min(1))
                max_values = np.maximum(max_values, X[0:3,:].max(1))
    
        self.min_values = min_values
        self.max_values = max_values
        
        #self.ScaleScene(ax)
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events() 
        
        return object_list #min_values.ravel(), max_values.ravel()    
    
    
    def UpdateObject(self, object_list = None, obj_name = 'base', obj_num = 1, obj_pose = 6*[None] ):
        # update the existing object with new extrinsic data
        # checks
        if object_list is None:
            object_list = self.object_list_current
        
        
        objects_list_found = [x for x in object_list if x.name == obj_name]
        if len(objects_list_found) < obj_num:
            self.Print('No object %s with number %s is found' %(obj_name,obj_num))
            return
        
        # objects are always 1,2,3
        k = obj_num - 1 
        
        # update not none
        for i,v in enumerate(obj_pose):
            if v is not None : 
                objects_list_found[k].extrinsics[i] = v

        objects_list_found = self.DrawObjects(self.ax, objects_list_found)
                
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

    
    def ConvertToExtrinsics(self, pose_list, use_inv = False):
        # converts 4x4 mat represenation to tvex and rvec
        extrinsics          = np.zeros((len(pose_list),6))
        for k in range(len(pose_list)):
            X           = pose_list[k]
            if use_inv:
                X = inv(X)
            #vvec, _     = cv.Rodrigues(X[0:3,0:3])
            vvec        = scirot.from_matrix(X[0:3,0:3]).as_rotvec()
            rvec        = scirot.from_rotvec(vvec.reshape((1,3))).as_euler('xyz',degrees=True).reshape((1,3))            
            tvec        = X[0:3,3].reshape((1,3))
            
            # assign
            pose        = np.hstack((tvec,rvec))
            extrinsics[k,:] = pose
            
        return extrinsics
        
    def ConertToExtrinsics(self, pose_list, use_inv = False):
        # converts 4x4 mat represenation to tvex and rvec
        extrinsics          = np.zeros((len(pose_list),6))
        for k in range(len(pose_list)):
            X           = pose_list[k]
            if use_inv:
                X = inv(X)
            #vvec, _     = cv.Rodrigues(X[0:3,0:3])
            vvec        = scirot.from_matrix(X[0:3,0:3]).as_rotvec()
            rvec        = scirot.from_rotvec(vvec.reshape((1,3))).as_euler('xyz',degrees=True).reshape((1,3))            
            tvec        = X[0:3,3].reshape((1,3))
            
            # assign
            pose        = np.hstack((tvec,rvec))
            extrinsics[k,:] = pose
            
        return extrinsics

    
    
#    def ShowSceneFromPose(self, cam_pose_list, grip_pose_list, pose_board, patternCentric = False):
#    # loads scene parameters to show the scene
#        square_size         = 20
#        board_width         = int(9)*square_size
#        board_height        = int(6)*square_size
#        cam_width           = 50
#        cam_height          = 40
#        cam_depth           = 70
#        #patternCentric      = True
#        
#        max_show            = np.minimum(2,len(cam_pose_list))
#        
#        # when pose is 6 vector
#        if type(pose_board) is list:
#            pose_board = [PoseToMatrix(pose_board[0])]
#            
#        
#        # convert camera to tvec and rvec
#        extrinsics          = self.ConertToExtrinsics(cam_pose_list[0:max_show], True)
#        params_cam          = {'extrinsics':extrinsics,'width':cam_width,'height':cam_height,'depth':cam_depth}
#
#        # convert gripper to tvec and rvec
#        extrinsics          = self.ConertToExtrinsics(grip_pose_list[0:max_show])
#        params_grip         = {'extrinsics':extrinsics,'width':cam_width/2,'height':cam_height,'depth':cam_depth}
#        
#        # board
#        extrinsics_board    = self.ConertToExtrinsics(pose_board) 
#        params_board        = {'extrinsics':extrinsics_board,'width':board_width,'height':board_height,'depth':0}
#        
#        params_obj          = None
#        
#        self.ShowSceneBoardCamObjGrip(params_cam, params_obj, params_board, params_grip, patternCentric)
#   
#        return True
    
    def Finish(self):
        
        #cv.destroyAllWindows()
        self.Print('Display Manager is cleared')
 
           
    def Print(self, txt='',level='I'):
        
        if level == 'I':
            ptxt = 'I: DM: %s' % txt
            logger.info(ptxt)  
        if level == 'W':
            ptxt = 'W: DM: %s' % txt
            logger.warning(ptxt)  
        if level == 'E':
            ptxt = 'E: DM: %s' % txt
            logger.error(ptxt)  
           
        print(ptxt)

    def TestScenePose(self, cfg):
        #    
        board_width         = int(9)
        board_height        = int(6)
        square_size         = cfg.GetSquareSize()
        scale_focal         = 0.03
        patternCentric      = False
        
        # extract params from config
        camMtrx, camDistortion,camResolution, camFocus, camId, camGain, camExposure = cfg.GetCameraParams()
        faces, labels, coords   = cfg.GetObjectParameters()
        folders                 = cfg.GetJSONFolders()
        files=[]
        for j in range(len(folders)):
            files += listdir(['.json'], folders[j])
        
        self.ShowScene(camMtrx, camResolution[0], camResolution[1], camDistortion, scale_focal, board_width, board_height, square_size, patternCentric, coords, files, labels, faces)
    
    def TestInitShowCameraGripperHangers(self, ax = None, fig = None):
        # actual use case from hangers manageer
        ax                  = self.InitScene(ax, fig)
        
        # board just for reference
        b1                  = np.array([-100.2, -100.0,  0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = self.GetBoardParams(extrinsics_board)
                
        # griper params    
        e1                  = np.array([754.2, 148.2, 700.1, -120.5, 0.7, -84.2 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1))    
        griper_list          = self.GetGripperParams(extrinsics_grip)
        
        # camera params from the gripper
        e1                  = np.array([654.2, 148.2, 800.1, -120.5, 0.7, -84.2 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1))    
        camera_list         = self.GetCameraParams(extrinsics_cam)

        # lines
        v1                  = np.array([1000.0,  400.0,     530.0,   90.0, 90.0, -0.0 ]).reshape(1,6)
        #v2                  = np.array([1000.0,  400.0,     540.0,   90.0, 90.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1))    
        line_list          = self.GetLineParams(extrinsics_obj)
        
        # hangers
        v1                  = np.array([1000.0, -100.0,     530.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([1000.0,  100.0,     535.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        hanger_list          = self.GetHangerParams(extrinsics_obj)

        # separators
        v1                  = np.array([1000.0,   -5.0,     535.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([1000.0,  150.0,     535.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        separator_list      = self.GetSeparatorParams(extrinsics_obj)


        patternCentric      = False
        object_list         = board_list + camera_list + griper_list + line_list + hanger_list + separator_list
        self.DrawObjects(ax, object_list, patternCentric)            
        self.ScaleScene(ax)
        
        # remember me
        self.object_list_current = object_list
        return object_list        
               
    def TestInitShowCameraBaseTool(self, ax = None, fig = None):
        # actual use case from hangers manageer
        ax                  = self.InitScene(ax, fig)
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        b2                  = np.array([300.0,   300.0,   100.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1,b2))
        b_radius            = [500, 200]
        base_list           = self.GetBaseParams(extrinsics_board, b_radius)
        
                
        # camera params    
        e1                  = np.array([1000.0,  1000.0,  800.0,   120.0, 70.0, 0 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1))    
        cam_list            = self.GetCameraParams(extrinsics_cam)


        e1                  = np.array([800.0,   300.0,    400.0,    0.0,  120.0,  0.0 ]).reshape(1,6)
        e2                  = np.array([500.0,   300.0,    200.0,  180.0, -90.0,  0 ]).reshape(1,6)
        e3                  = np.array([700.0,   400.0,    400.0 , 180.0,   20.0, 0.0 ]).reshape(1,6)
        extrinsics_obj     = np.vstack((e3)) #,e2,e3))    
        tool_list         = self.GetToolParams(extrinsics_obj)
        
        # points
        e1                  = np.array([300.2, 200.2, 100.1,  ]).reshape(1,3)
        e2                  = np.array([300.2, 800.2, 100.1,  ]).reshape(1,3)
        e3                  = np.array([200.2, 300.2, 100.1,  ]).reshape(1,3)
        e4                  = np.array([800.2, 300.2, 100.1,  ]).reshape(1,3)
        extrinsics_point    = np.vstack((e1,e2,e3,e4))    
        point_list          = self.GetPointParams(extrinsics_point)
        

        patternCentric      = False
        object_list         = base_list +  tool_list + cam_list + point_list
        self.DrawObjects(ax, object_list, patternCentric) 
        self.ScaleScene(ax)   
        
        # remember me
        self.object_list_current = object_list

        # refresh
        #fig.canvas.draw()
        
        return object_list
    
    def TestTrafficCamera(self, ax = None, fig = None):
        # actual use case from singapore
        ax                  = self.InitScene(ax, fig)
        
        # board just for reference
#        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
#        extrinsics_board    = np.vstack((b1))
#        b_radius            = [500]
#        base_list           = self.GetBaseParams(extrinsics_board, b_radius)
        
        bSize               = (2000,8000)
        b1                  = np.array([-bSize[0]/2, 0.0, 0.0, 0.0, 0.0, 0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = self.GetBoardParams(extrinsics_board, sizeBB = bSize)        
        
        # camera params    
        cSize               = (200,300,400)
        e1                  = np.array([0.0,  0.0, 3000.0,  -120.0, 0.0, 0.0 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1))    
        cam_list            = self.GetCameraParams(extrinsics_cam, sizeBB = cSize)

        bSize               = (200,500,300)
        e1                  = np.array([-500.0,   7000.0,    0.0,  0.0,  0.0,  0.0 ]).reshape(1,6)
        e2                  = np.array([-500.0,   6000.0,    0.0,  0.0,  0.0,  0.0 ]).reshape(1,6)
        e3                  = np.array([ 400.0,   2000.0,    0.0 , 0.0,  0.0,  0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((e1,e2,e3))    
        box_list           = self.GetBoxParams(extrinsics_obj, sizeBB = bSize)
        
#        # points
#        e1                  = np.array([300.2, 200.2, 100.1,  ]).reshape(1,3)
#        e2                  = np.array([300.2, 800.2, 100.1,  ]).reshape(1,3)
#        e3                  = np.array([200.2, 300.2, 100.1,  ]).reshape(1,3)
#        e4                  = np.array([800.2, 300.2, 100.1,  ]).reshape(1,3)
#        extrinsics_point    = np.vstack((e1,e2,e3,e4))    
#        point_list          = self.GetPointParams(extrinsics_point)
        

        patternCentric      = False
        object_list         = board_list +  box_list + cam_list #+ point_list
        self.DrawObjects(ax, object_list, patternCentric) 
        self.ScaleScene(ax)   
        
        # remember me
        self.object_list_current = object_list

        return object_list    
    
    
   
    def TestInitShowCameraGripperPoints(self, ax = None, fig = None):
        # actual use case from hangers manageer
        ax                  = self.InitScene(ax, fig)
        
        # board just for reference
        b1                  = np.array([834.2, 0.0,  -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = self.GetBoardParams(extrinsics_board)
                
        # griper params    
        g1                  = np.array([854.2, 148.2, -13.1, -177.5, 53.7, -84.2 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((g1))    
        griper_list          = self.GetGripperParams(extrinsics_grip)
        
        # camera params from the gripper
        c1                  = np.array([854.2, 68.2, 93.1, -177.5, 53.7, -84.2 ]).reshape(1,6)
        extrinsics_cam     = np.vstack((c1))          
        camera_list         = self.GetCameraParams(extrinsics_cam)
        
        # lines
        v1                  = np.array([500.0,  500.0,     40.0,   90.0, 0.0, 90.0 ]).reshape(1,6)
        v2                  = np.array([500.0,  500.0,    -40.0,   90.0, 0.0, 90.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        line_list          = self.GetLineParams(extrinsics_obj)        

        # points
#        e1                  = np.array([700.2, 280.2, 100.1,  ]).reshape(1,3)
#        e2                  = np.array([750.2, 300.2, 10.1,  ]).reshape(1,3)
#        e3                  = np.array([770.2, 330.2, 1.1,  ]).reshape(1,3)
#        e4                  = np.array([800.2, 370.2, 100.1,  ]).reshape(1,3)
#        extrinsics_point    = np.vstack((e1,e2,e3,e4))    
#        point_list          = self.GetPointParams(extrinsics_point)
        point_num           = 20
        c1                  = np.round(c1).ravel()
        ex                  = np.random.randint(c1[0]-200, c1[0]+200, size = (point_num,1))
        ey                  = np.random.randint(c1[1]+300, c1[1]+500, size = (point_num,1))
        ez                  = np.random.randint(c1[2]-300, c1[2]-0,   size = (point_num,1))
        extrinsics_point    = np.hstack((ex,ey,ez))    
        point_list          = self.GetPointParams(extrinsics_point)

        patternCentric      = False
        object_list         = board_list + camera_list + griper_list + point_list + line_list
        self.DrawObjects(ax, object_list, patternCentric)            
        self.ScaleScene(ax) 
        ax.grid(False)
    
    def TestRenderaGripperHangersSeparators(self, ax, hanger_pose = [], separator_pose = []):
        # actual use case from hangers manageer
        if len(hanger_pose) < 1:
            hanger_pose = [-100, 100]
            
        if len(separator_pose) < 1:
            separator_pose = [-10, 150]

        # board just for reference
        b1                  = np.array([0.0, 834.2, -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = self.GetBoardParams(extrinsics_board)
        
        # gripper params    
        e1                  = np.array([148.2, 854.2, 30.1, -177.5, 80.7, -84.2 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1))    
        params_grip         = self.GetGripperParams(extrinsics_grip)
        
        # convert camera params   
        params_cam          = self.GetCameraParams()
        e1                  = np.array([148.2, 754.2, 80.1, -177.5, 80.7, -84.2 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1))    
        params_cam          = {'extrinsics':extrinsics_cam,'width':40,'height':30,'depth':50, 'name':'cameras'}

        
        # lines
        v1                  = np.array([-200.0,  1000.0,   30.0,   0.0, 90.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([-200.0,  1000.0,   35.0,   0.0, 90.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        params_line          = {'extrinsics':extrinsics_obj,'width':1,'height':500, 'depth':10,'name': 'lines'}
        
        # hangers
        extrinsics_obj      = np.zeros((0,6))
        for k in range(len(hanger_pose)):
            v1                  = np.array([hanger_pose[k],  1000.0,   30.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
            extrinsics_obj      = np.vstack((extrinsics_obj,v1))    
        params_hanger       = {'extrinsics':extrinsics_obj,'width':1,'height':30, 'depth':10,'name': 'hangers'}

        # separators
        extrinsics_obj      = np.zeros((0,6))
        for k in range(len(separator_pose)):
            v1                  = np.array([separator_pose[k],    1000.0,   35.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
            extrinsics_obj      = np.vstack((extrinsics_obj,v1))    
        params_saparator    = {'extrinsics':extrinsics_obj,'width':1,'height':30, 'depth':20,'name': 'separators'}



        patternCentric      = False
        # params_rays is a list - need to be merged
        params_list         = [params_cam, params_board, params_grip, params_line, params_hanger, params_saparator]
        isOk                = self.ShowSceneModelList(params_list, patternCentric, ax)        
        
#%% --------------------------           
class TestDisplayManager(unittest.TestCase): 
               
    def test_Create(self):
        cfg         = ConfigManager(r'D:\RobotAI\Customers\ITPAero\Objects\Trajectory_Stereo_01')
        d           = DisplayManager(cfg)
        self.assertEqual(0, len(d.prevImage))
        d.Finish()
        
    def test_ShowBoard(self):
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()

        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        b2                  = np.array([0.0,   0.0,   10.0,  0.0,  0.0, -90.0 ]).reshape(1,6)    
        b3                  = np.array([0.0,   0.0,   0.0,  0.0,  45.0, 0.0 ]).reshape(1,6) 
        extrinsics_board    = np.vstack((b1,b2,b3))
      
        #import argparse
        board_list          = d.GetBoardParams(extrinsics_board)
        

        patternCentric      = False
        d.DrawObjects(ax, board_list, patternCentric)
        d.Finish()
        #self.assertEqual(isOk, True)
        
        
    def test_ShowCamera(self):
        # test camera placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                
        # camera params    
        e1                  = np.array([200.0,  0.0,    500.0,    0.0,  0.0, 90.0 ]).reshape(1,6)
        e2                  = np.array([0.0,  500.0,  200.0,  180.0, -0.0, 0 ]).reshape(1,6)
        e3                  = np.array([0.0,  100.0,  400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1,e2,e3))    
        camera_list         = d.GetCameraParams(extrinsics_cam)
        

        patternCentric      = False
        object_list         = board_list + camera_list
        d.DrawObjects(ax, object_list, patternCentric)
        #self.assertEqual(isOk, True)
        
    def test_ShowGripper(self):
        # test gripper placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                
        # griper params    
        e1                  = np.array([200.0,  0.0,    500.0,    0.0,  0.0, 90.0 ]).reshape(1,6)
        e2                  = np.array([0.0,  500.0,  200.0,  180.0, -0.0, 0 ]).reshape(1,6)
        e3                  = np.array([0.0,  100.0,  400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    
        griper_list          = d.GetGripperParams(extrinsics_grip)
        
        # camera params from the gripper
        camera_list         = d.GetCameraParams(None)


        patternCentric      = False
        object_list         = board_list + camera_list + griper_list
        d.DrawObjects(ax, object_list, patternCentric)
        #self.assertEqual(isOk, True)
        
    def test_ShowPoints(self):
        # test point placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                
        # griper params    
        e1                  = np.array([200.0,  0.0,    500.0,    0.0,  0.0, 90.0 ]).reshape(1,6)
        e2                  = np.array([0.0,  500.0,  200.0,  180.0, -0.0, 0 ]).reshape(1,6)
        e3                  = np.array([0.0,  100.0,  400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    
        griper_list          = d.GetGripperParams(extrinsics_grip)
        
        # camera params from the gripper
        camera_list         = d.GetCameraParams(None)
        
        # points
        e1                  = np.array([148.2, 334.2, 200.1,  ]).reshape(1,3)
        e2                  = np.array([158.2, 434.2, 100.1,  ]).reshape(1,3)
        e3                  = np.array([168.2, 234.2, 300.1,  ]).reshape(1,3)
        extrinsics_point    = np.vstack((e1,e2,e3))    
        point_list          = d.GetPointsParams(extrinsics_point)


        patternCentric      = False
        object_list         = board_list + camera_list + griper_list + point_list
        d.DrawObjects(ax, object_list, patternCentric)
        #self.assertEqual(isOk, True)        
                
        
    def test_ShowLines(self):
        # test line placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                
        # griper params    
        e1                  = np.array([0.0,  0.0,  200.0,  180.0, -0.0, 0 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1))    
        griper_list          = d.GetGripperParams(extrinsics_grip)
        
        # camera params from the gripper
        camera_list         = d.GetCameraParams(None)

        
        v1                  = np.array([10.0,  10.0,   0.0,   0.0, -0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([50.0,  50.0,  0.0,  45.0, -0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        line_list          = d.GetLineParams(extrinsics_obj)


        patternCentric      = False
        object_list         = board_list + camera_list + griper_list + line_list
        d.DrawObjects(ax, object_list, patternCentric)
        
    def test_ShowHangers(self):
        # test line placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                

        e1                  = np.array([1200.0,  -300.0,    400.0,    0.0,  0.0,  90.0 ]).reshape(1,6)
        e2                  = np.array([1000.0,  300.0,    200.0,  180.0, -0.0,  0 ]).reshape(1,6)
        e3                  = np.array([1000.0,  400.0,    400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_obj     = np.vstack((e1,e2,e3))    
        hanger_list         = d.GetHangerParams(extrinsics_obj)


        patternCentric      = False
        object_list         = board_list +  hanger_list
        d.DrawObjects(ax, object_list, patternCentric) 
        d.ScaleScene(ax)
        
    def test_ShowSeparators(self):
        # test line placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                

        e1                  = np.array([200.0,  300.0,    400.0,    0.0,  0.0,  90.0 ]).reshape(1,6)
        e2                  = np.array([400.0,  300.0,    200.0,  180.0, -0.0,  0 ]).reshape(1,6)
        e3                  = np.array([300.0,  100.0,    400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_obj     = np.vstack((e1,e2,e3))    
        separator_list         = d.GetSeparatorParams(extrinsics_obj)


        patternCentric      = False
        object_list         = board_list +  separator_list
        d.DrawObjects(ax, object_list, patternCentric)     
        
    def test_ShowBase(self):
        # test line placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        b2                  = np.array([300.0,   300.0,   100.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1,b2))
        b_radius            = [500, 200]
        base_list          = d.GetBaseParams(extrinsics_board, b_radius)
        
        # make smaller radius
        base_list[1].params_model['width'] = 300
        base_list[1].create_model()
                
        # camera params    
        e1                  = np.array([1200.0,  1200.0,  800.0,   90.0, 70.0, -45 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1))    
        cam_list            = d.GetCameraParams(extrinsics_cam)


        e1                  = np.array([800.0,  -300.0,    400.0,    0.0,  0.0,  90.0 ]).reshape(1,6)
        e2                  = np.array([1000.0,  300.0,    200.0,  180.0, -0.0,  0 ]).reshape(1,6)
        e3                  = np.array([1000.0,  400.0,    400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_obj     = np.vstack((e1,e2,e3))    
        hanger_list         = d.GetToolParams(extrinsics_obj)


        patternCentric      = False
        object_list         = base_list +  hanger_list + cam_list
        d.DrawObjects(ax, object_list, patternCentric) 
        d.ScaleScene(ax)        
        
        
        
    def test_ShowTwoCamerasAndLines(self):
        # test camera placement
        cfg                 = None
        d                   = DisplayManager(cfg)
        
        # import argparse
        square_size         = 10
        board_width         = int(9)*square_size
        board_height        = int(6)*square_size
        
        # import argparse
        cam_width           = 20
        cam_height          = 15
        cam_depth           = 20
        
        # import argparse
        obj_width           = 40
        obj_height          = 400
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = {'extrinsics':extrinsics_board,'width':board_width,'height':board_height,'depth':0}
                
        # camera params    
        e1                  = np.array([30.0,  200.0,  200.0,   90.0, 0.0, -10 ]).reshape(1,6)
        e2                  = np.array([-30.0,  200.0,  200.0,  90.0, 0.0, 10 ]).reshape(1,6)
        extrinsics_cam      = np.vstack((e1,e2))    
        params_cam          = {'extrinsics':extrinsics_cam,'width':cam_width,'height':cam_height,'depth':cam_depth}
        
        v1                  = np.array([10.0,  10.0,   0.0,   0.0, -0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([50.0,  50.0,  0.0,  45.0, -0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        params_obj          = {'extrinsics':extrinsics_obj,'width':obj_width,'height':obj_height}
        
        patternCentric      = False
        isOk                = d.ShowSceneNew(params_cam, params_obj, params_board, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)
        
        
        
    def test_ShowCameraAndPixelRay(self):
        # test gripper placement
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\ITPAero\Objects\TurbineStereo_01')
        d                   = DisplayManager(cfg)
        d.Init()
        
        # init important  params
        rvecGC, tvecGC      = cfg.GetCameraGripperTransform()
        poseGC              = cfg.TransformRvecTvecToPose(rvecGC, tvecGC,'xyz')
        mtrxGC              = PoseToMatrix(poseGC)
        #mtrxCG              = np.inv(mtrxGC)
        cx, cy              = d.camMatrix[0,2], d.camMatrix[1,2]
        
        #import argparse
        cam_width           = 40
        cam_height          = 30
        cam_depth           = 90
        
        # import argparse
        square_size         = 10
        board_width         = int(9)*square_size
        board_height        = int(6)*square_size        
        
        # board just for reference
        b1                  = np.array([158.2, 834.2, 0.1,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = {'extrinsics':extrinsics_board,'width':board_width,'height':board_height,'depth':10, 'name':'board'}
        
        
        # gripper params    
        e1                  = np.array([148.2, 834.2, -13.1, -177.5, 13.7, -84.2 ]).reshape(1,6)
        e2                  = np.array([158.2, 834.2, -13.1,  172.2, 13.3, -79.4 ]).reshape(1,6)
        e3                  = np.array([168.2, 834.2, -13.1,  -166.5, 8.0, -80.5 ]).reshape(1,6)
        #extrinsics_grip     = np.vstack((e1,e2,e3))    
        e1                  = np.array([148.2, 844.2, 100.0, -180, 0, 0 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1))    

        params_grip         = {'extrinsics':extrinsics_grip,'width':20,'height':40,'depth':40}
        
        # convert camera params   
        extrinsics_cam      = np.zeros((0,6))
        for k in range(len(extrinsics_grip)):
            poseBG          = extrinsics_grip[k]
            mtrxBG          = PoseToMatrix(poseBG)
            mtrxBC          = mtrxBG #np.dot(mtrxBG, mtrxGC)
            poseBC          = MatrixToPose(mtrxBC)
            extrinsics_cam  = np.vstack((extrinsics_cam,poseBC))
              
        params_cam          = {'extrinsics':extrinsics_cam,'width':cam_width,'height':cam_height,'depth':cam_depth}
        
        # convert rays
        sensorScale         = 200
        camMtrxInv          = inv(d.camMatrix)   
        rayPixels           = np.array([[110]+cx,[0]+cy,[1]]) # x,y,z in pixels camera
        rayCoords           = np.dot(camMtrxInv, rayPixels)*sensorScale
        obj_width, obj_height, obj_depth   = rayCoords[0,:], rayCoords[1,:], rayCoords[2,:]        
        params_obj          = {'extrinsics':extrinsics_cam,'width':obj_width,'height':obj_height, 'depth':obj_depth, 'name':'camera_ray'}        

        #params_grip          = None
        params_grip          = None
        patternCentric      = False
        isOk                = d.ShowSceneBoardCamObjGrip(params_cam, params_obj, params_board, params_grip, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)           
        
        
    def test_RealTimeUpdate(self):
        # test figure update
        cfg                 = None
        d                   = DisplayManager(cfg)
        ax                  = d.InitScene()
        
        # board just for reference
        b1                  = np.array([0.0,   0.0,   0.0,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        board_list          = d.GetBoardParams(extrinsics_board)
                
        # griper params    
        e1                  = np.array([200.0,  0.0,    500.0,    0.0,  0.0, 90.0 ]).reshape(1,6)
        e2                  = np.array([0.0,  500.0,  200.0,  180.0, -0.0, 0 ]).reshape(1,6)
        e3                  = np.array([0.0,  100.0,  400.0 , 180.0,  0.0, -90.0 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    
        camera_list         = d.GetCameraParams(extrinsics_grip)
        
        patternCentric      = False
        object_list         = board_list + camera_list
        object_list         = d.DrawObjects(ax, object_list, patternCentric)
        
        #line1           = np.random.uniform(low=0.5, high=130.3, size=(2,3))
        line1           = np.arange(400)+50
        for k in range(len(line1)):
            
            for j in range(len(object_list)):
                if object_list[j].name == 'board':
                    continue
                object_list[j].extrinsics[1]       = line1[k] 
                
            object_list = d.DrawObjects(ax, object_list)
                    
            d.fig.canvas.draw()
            d.fig.canvas.flush_events()
        
            plt.pause(0.1)

            
        d.Finish()
        
    def test_RenderCameraCalibrationScene(self):
        # test figure update
        cfg                 = None
        d                   = DisplayManager(cfg)
        
        # tvec, rvec
        tvecL, rvecL, tvecR, rvecR = [], [], [], []
                
        # camera params    
        tvecL.append(np.array([30.0,  200.0,  200.0]).reshape(1,3))
        rvecL.append(fromEulerToRvec(np.array([ 90.0, 0.0, -10 ]).reshape(3,1)))
        tvecR.append(np.array([-30.0,  200.0,  200.0]).reshape(1,3))
        rvecR.append(fromEulerToRvec(np.array([ 90.0, 0.0,  10 ]).reshape(3,1)))
        
        isOk                = d.RenderCameraCalibrationScene(rvecL, tvecL, rvecR, tvecR)
                    
        d.Finish()
        self.assertEqual(isOk, True)         
        
    def test_ShowSceneFromPose(self):
        # transfer pose data from standard pose
        cfg                 = None
        d                   = DisplayManager(cfg)
        
        # tvec, rvec
        cam_pose_list, grip_pose_list, pose_board_list = [], [], []
                
        # camera params    
        tvec   = np.array([0.0,  0.0,  100.0]).ravel()
        rvec   = fromEulerToRvec(np.array([ 0.0, 0.0,  0 ]).reshape(1,3)).ravel()
        cam_pose_list.append(fromRvecTvecToHomogenious(rvec,tvec))
        rvec   = fromEulerToRvec(np.array([ 0.0, 0.0, -90 ]).reshape(1,3)).ravel()
        cam_pose_list.append(fromRvecTvecToHomogenious(rvec,tvec))
        
        tvec   = np.array([50.0,  50.0,  100.0]).ravel()
        rvec   = fromEulerToRvec(np.array([ 0.0, 0.0,  0 ]).reshape(1,3)).ravel()
        grip_pose_list.append(fromRvecTvecToHomogenious(rvec,tvec))
        
        tvec   = np.array([0.0, 0.0, 0.0]).ravel()
        rvec   = fromEulerToRvec(np.array([ 0.0, 0.0, 0.0 ]).reshape(1,3)).ravel()
        pose_board_list.append(fromRvecTvecToHomogenious(rvec,tvec))
        
        patternCentric = True
        isOk   = d.ShowSceneFromPose(cam_pose_list, grip_pose_list, pose_board_list, patternCentric)
                    
        d.Finish()
        self.assertEqual(isOk, True)         
        
    
        
    def test_ShowCameraGripperAndPixelRays(self):
        # actual use case from stereo manageer
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\ITPAero\Objects\Trajectory_Stereo_01')
        d                   = DisplayManager(cfg)
        
        # init important  params
        mtrxGC              = PoseToMatrix(d.poseGC)
        cx, cy              = d.camMatrix[0,2], d.camMatrix[1,2]
        
        #import argparse
        cam_width           = 40
        cam_height          = 30
        cam_depth           = 90
        
        # import argparse
        square_size         = 10
        board_width         = int(9)*square_size
        board_height        = int(6)*square_size        
        
        # board just for reference
        b1                  = np.array([158.2, 834.2, -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = {'extrinsics':extrinsics_board,'width':board_width,'height':board_height,'depth':10, 'name':'boards'}
        
        
        # gripper params    
        e1                  = np.array([148.2, 854.2, -13.1, -177.5, 13.7, -84.2 ]).reshape(1,6)
        e2                  = np.array([158.2, 834.2, -13.1,  172.2, 13.3, -79.4 ]).reshape(1,6)
        e3                  = np.array([168.2, 814.2, -13.1,  -166.5, 8.0, -80.5 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    

        params_grip         = {'extrinsics':extrinsics_grip,'width':20,'height':40,'depth':40, 'name':'grippers'}
        
        # convert camera params   
        extrinsics_cam      = np.zeros((0,6))
        for k in range(len(extrinsics_grip)):
            poseBG          = extrinsics_grip[k]
            mtrxBG          = PoseToMatrix(poseBG)
            mtrxBC          = np.dot(mtrxBG, mtrxGC)
            poseBC          = MatrixToPose(mtrxBC)
            extrinsics_cam  = np.vstack((extrinsics_cam,poseBC))
              
        params_cam          = {'extrinsics':extrinsics_cam,'width':cam_width,'height':cam_height,'depth':cam_depth, 'name':'cameras'}
        
        # convert rays
        sensorScale         = 200
        camMtrxInv          = inv(d.camMatrix)   
        rayPixels           = np.array([[110]+cx,[0]+cy,[1]]) # x,y,z in pixels camera
        rayCoords           = np.dot(camMtrxInv, rayPixels)*sensorScale
        obj_width, obj_height, obj_depth   = rayCoords[0,:], rayCoords[1,:], rayCoords[2,:]        
        params_obj          = {'extrinsics':extrinsics_cam,'width':obj_width,'height':obj_height, 'depth':obj_depth, 'name':'rays'}        

        #params_grip          = None
        #params_grip          = None
        patternCentric      = False
        isOk                = d.ShowSceneBoardCamObjGrip(params_cam, params_obj, params_board, params_grip, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)           
        
    def test_ShowCameraGripperAndPixelRaysUsingList(self):
        # actual use case from stereo manageer
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\ITPAero\Objects\Trajectory_Stereo_01')
        d                   = DisplayManager(cfg)
        
        # init important  params
        cx, cy              = d.camMatrix[0,2], d.camMatrix[1,2]
   
        # board just for reference
        b1                  = np.array([158.2, 834.2, -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = d.GetBoardParams(extrinsics_board)
        
        # gripper params    
        e1                  = np.array([148.2, 854.2, -13.1, -177.5, 13.7, -84.2 ]).reshape(1,6)
        e2                  = np.array([158.2, 834.2, -13.1,  172.2, 13.3, -79.4 ]).reshape(1,6)
        e3                  = np.array([168.2, 814.2, -13.1,  -166.5, 8.0, -80.5 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    
        params_grip         = d.GetGripperParams(extrinsics_grip)
        
        # convert camera params   
        params_cam          =  d.GetCameraParams()
        
        # convert rays
        p1                  = np.array([[0 + cx, 0 + cy]]) # x,y,z in pixels camera 1
        p2                  = np.array([[200 + cx, 50 + cy]]) # x,y,z in pixels camera 1
        p3                  = np.array([[-50 + cx, -50 + cy]]) # x,y,z in pixels camera 1
        
        rayPixels           = [p1,p2,p3]
        params_rays         = d.GetRaysParams(rayPixels)   

        #params_grip          = None
        #params_grip          = None
        patternCentric      = False
        # params_rays is a list - need to be merged
        params_list         = [params_cam, params_board, params_grip] + params_rays
        isOk                = d.ShowSceneModelList(params_list, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)           

    def test_ShowCameraGripperPoints(self):
        # actual use case from stereo manageer
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\ITPAero\Objects\TrajectoryStereo_01')
        d                   = DisplayManager(cfg)
        

        # board just for reference
        b1                  = np.array([158.2, 834.2, -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = d.GetBoardParams(extrinsics_board)
        
        # gripper params    
        e1                  = np.array([148.2, 854.2, -13.1, -177.5, 13.7, -84.2 ]).reshape(1,6)
        e2                  = np.array([158.2, 834.2, -13.1,  172.2, 13.3, -79.4 ]).reshape(1,6)
        e3                  = np.array([168.2, 814.2, -13.1,  -166.5, 8.0, -80.5 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1,e2,e3))    
        params_grip         = d.GetGripperParams(extrinsics_grip)
        
        # convert camera params   
        params_cam          =  d.GetCameraParams()
        
        # points
        p1                  = np.array([158.2, 834.2, -220]).reshape(1,3) # x,y,z in mm 
        p2                  = np.array([160.0, 900.0, 0.0]).reshape(1,3) # x,y,z in mm 
        p3                  = np.array([148.2, 854.2, -31.1]).reshape(1,3) # x,y,z in mm 
        
        points3D           = np.vstack((p1,p2,p3)) 
        params_points       = d.GetPointParams(points3D)   

        #params_grip          = None
        #params_grip          = None
        patternCentric      = False
        # params_rays is a list - need to be merged
        params_list         = [params_cam, params_board, params_grip] + params_points
        isOk                = d.ShowSceneModelList(params_list, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)    
        
    def test_ShowCameraGripperHangers(self):
        # actual use case from hangers manageer
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\Inditex\Objects\Inditex-Hanger-02')
        d                   = DisplayManager(cfg)
        

        # board just for reference
        b1                  = np.array([0.0, 834.2, -200,  0.0,  0.0, -0.0 ]).reshape(1,6)
        extrinsics_board    = np.vstack((b1))
        params_board        = d.GetBoardParams(extrinsics_board)
        
        # gripper params    
        e1                  = np.array([148.2, 854.2, -13.1, -177.5, 13.7, -84.2 ]).reshape(1,6)
        extrinsics_grip     = np.vstack((e1))    
        params_grip         = d.GetGripperParams(extrinsics_grip)
        
        # convert camera params   
        params_cam          =  d.GetCameraParams()
        
        # lines
        v1                  = np.array([-200.0,  1000.0,   30.0,   0.0, 90.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([-200.0,  1000.0,   35.0,   0.0, 90.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        params_line          = {'extrinsics':extrinsics_obj,'width':1,'height':500, 'depth':10,'name': 'lines'}
        
        # hangers
        v1                  = np.array([-100.0,  1000.0,   30.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([ 100.0,  1000.0,   35.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        params_hanger       = {'extrinsics':extrinsics_obj,'width':1,'height':30, 'depth':10,'name': 'hangers'}

        # separators
        v1                  = np.array([-5.0,    1000.0,   35.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        v2                  = np.array([ 150.0,  1000.0,   35.0,   0.0, 0.0, -0.0 ]).reshape(1,6)
        extrinsics_obj      = np.vstack((v1,v2))    
        params_saparator    = {'extrinsics':extrinsics_obj,'width':1,'height':30, 'depth':20,'name': 'separators'}



        patternCentric      = False
        # params_rays is a list - need to be merged
        params_list         = [params_cam, params_board, params_grip, params_line, params_hanger, params_saparator]
        isOk                = d.ShowSceneModelList(params_list, patternCentric)
        d.Finish()
        self.assertEqual(isOk, True)  
        
    def test_TestInitShowCameraGripperHangers(self):
        # a use case
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\Inditex\Objects\Inditex-Hanger-02')
        d                   = DisplayManager(cfg)        
        d.TestInitShowCameraGripperHangers()
        
        
    def test_TestRenderaGripperHangersSeparators(self):
        # actual use case from hangers manageer
        cfg                 = ConfigManager(r'D:\RobotAI\Customers\Inditex\Objects\Inditex-Hanger-02')
        d                   = DisplayManager(cfg)
        hanger_pose         = [-100, 100]
        separator_pose      = [-200, 200, 0]
        d.TestRenderaGripperHangersSeparators(None, hanger_pose, separator_pose)
        
    def test_TestInitShowCameraBaseTool(self):
        # actual use case from hangers manageer
        cfg                 = None #ConfigManager(r'D:\RobotAI\Customers\Inditex\Objects\Inditex-Hanger-02')
        d                   = DisplayManager(cfg)
        d.TestInitShowCameraBaseTool() 
        
    def test_TestTrafficCamera(self):
        # Singapore traffic management system
        cfg                 = None #ConfigManager(r'D:\RobotAI\Customers\Inditex\Objects\Inditex-Hanger-02')
        d                   = DisplayManager(cfg)
        d.TestTrafficCamera()         
        
        
    def test_RealTimeToolUpdate(self):
        # test figure update
        cfg                 = None
        d                   = DisplayManager(cfg)
        object_list         = d.TestInitShowCameraBaseTool()
        ax                  = d.ax
        #plt.show(block=False)
        
        #line1           = np.random.uniform(low=0.5, high=130.3, size=(2,3))
        line1           = np.arange(100)*4+50
        for k in range(len(line1)):
            
            for j in range(len(object_list)):
                if object_list[j].name == 'tool':
                    continue
                object_list[j].extrinsics[1]       = line1[k] 
                
            object_list = d.DrawObjects(ax, object_list)
                    
            d.fig.canvas.draw()
            d.fig.canvas.flush_events()
        
            plt.pause(0.1)
            #plt.show(block=False)

        d.Finish()
        
    def test_RealTimeToolUpdateWithFunction(self):
        # test figure update
        cfg                 = None
        d                   = DisplayManager(cfg)
        object_list         = d.TestInitShowCameraBaseTool()

        
        #line1           = np.random.uniform(low=0.5, high=130.3, size=(2,3))
        line1           = np.arange(400)+50
        tool_pose       = 6*[None]
        for k in range(len(line1)):
            
            tool_pose[1] = line1[k]
            d.UpdateObject(object_list, obj_name = 'tool', obj_num = 1, obj_pose = tool_pose )

        d.Finish()   
        
    def test_RealTimeCameraUpdateWithFunction(self):
        # test figure update - show camera and gripper motion
        cfg                 = None
        d                   = DisplayManager(cfg)
        object_list         = d.TestInitShowCameraGripperHangers()

        #line1           = np.random.uniform(low=0.5, high=130.3, size=(2,3))
        line1               = np.arange(400)+50
        grip_pose           = 6*[None]
        for k in range(len(line1)):
            
            grip_pose[1] = line1[k]
            d.UpdateObject(object_list, obj_name = 'gripper', obj_num = 1, obj_pose = grip_pose )
            d.UpdateObject(object_list, obj_name = 'camera',  obj_num = 1, obj_pose = grip_pose )

        d.Finish()             
        
        
        

#%% 
if __name__ == '__main__':
    print(__doc__)
#    cfg         = ConfigManager(r'D:\RobotAI\Customers\RobotAI\MouseBox-24Labels')
#    d           = DisplayManager(cfg)
#    #d.TestScene(cfg)
#    d.TestSceneNew()
#    d.Finish()
    
    # single test
    singletest = unittest.TestSuite()
    #singletest.addTest(TestDisplayManager("test_Create"))
    #singletest.addTest(TestDisplayManager("test_ShowBoard")) # ok
    #singletest.addTest(TestDisplayManager("test_ShowCamera")) # ok
    #singletest.addTest(TestDisplayManager("test_ShowGripper"))  # ok
    #singletest.addTest(TestDisplayManager("test_ShowPoints")) # on  
    #singletest.addTest(TestDisplayManager("test_ShowLines")) # ok 
    #singletest.addTest(TestDisplayManager("test_ShowHangers")) # ok
    #singletest.addTest(TestDisplayManager("test_ShowSeparators")) # ok
    #singletest.addTest(TestDisplayManager("test_ShowBase")) # ok
    

    #singletest.addTest(TestDisplayManager("test_RenderCameraCalibrationScene"))
    #singletest.addTest(TestDisplayManager("test_ShowSceneFromPose")) # nok
    #singletest.addTest(TestDisplayManager("test_ShowCameraAndGripper"))
    #singletest.addTest(TestDisplayManager("test_ShowCameraAndPixelRay"))
    #singletest.addTest(TestDisplayManager("test_ShowCameraGripperAndPixelRays"))    
    #singletest.addTest(TestDisplayManager("test_ShowCameraGripperAndPixelRaysUsingList"))   # ok 
    #singletest.addTest(TestDisplayManager("test_ShowCameraGripperPoints"))   # nok 
    #singletest.addTest(TestDisplayManager("test_ShowCameraGripperHangers"))   # ok 
    #singletest.addTest(TestDisplayManager("test_TestInitShowCameraGripperHangers"))   #  ok
    #singletest.addTest(TestDisplayManager("test_TestRenderaGripperHangersSeparators"))   #  
    #singletest.addTest(TestDisplayManager("test_TestInitShowCameraBaseTool"))   # ok 
    
    #singletest.addTest(TestDisplayManager("test_RealTimeUpdate"))  # ok    
    singletest.addTest(TestDisplayManager("test_RealTimeToolUpdate"))  # ok    
    #singletest.addTest(TestDisplayManager("test_RealTimeToolUpdateWithFunction"))  # ok    
    #singletest.addTest(TestDisplayManager("test_RealTimeCameraUpdateWithFunction"))  #  
    
    #singletest.addTest(TestDisplayManager("test_TestTrafficCamera")) 
    
    
    unittest.TextTestRunner().run(singletest)
    