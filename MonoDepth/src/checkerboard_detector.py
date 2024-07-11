# -*- coding: utf-8 -*-
#cython: language_level=3
#distutils: language=c++

"""
Created on Sep 04 16:53:13 2019
Debug module that simulates detection

-----------------------------
 Ver    Date     Who    Descr
-----------------------------
0101   09.07.24 UD     Started chess. 
-----------------------------
"""


import cv2
import numpy as np
import logging
import glob



class ObjectChess:
    """ 
    The object part that need to be detected and estimate its pose
    """

    def __init__(self):

        
        # check licenses
        self.name          = 'chess'
        self.pattern_size   = (9,6)
        self.square_size    = 20
        self.scale_factor   = 1

        self.pattern_points = []
        self.init_pattern(self.pattern_size, self.square_size)
        
        self.camera_matrix  = []
        self.dist_coeffs    = []
        

    def init(self):
        """

        """

        self.init_pattern(pattern_size = self.pattern_size, square_size = self.square_size)
        self.camera_matrix  = np.array([[1000,0,640],[0,1000,360],[0,0,1]])
        self.dist_coeffs    = np.array([0,0,0,0,0])

        
    def detect_objects(self, rgb_image):
        """
        
        """
        
        # select object to work with
        objects = {'objectId': [], 'rvecAll': [], 'tvecAll': [], 'objectQ': []}
        
        if len(self.camera_matrix)<1:
            self.Print('Load-update chess camera matrix','E')
            return objects
         
        found, corners      = self.find_corners(rgb_image)
        if not found:
            #self.Print("Failed to find corners in img" )
            return objects
        
        # in case that image is scaled
        corners             = corners/self.scale_factor
        
        rvec, tvec          = self.get_object_pose(self.pattern_points, corners, self.camera_matrix, self.dist_coeffs)        
        
        #self.res.res        = []    
        self.res.objects               = corners  # debug chess pattern
        
        objects['objectId']            = [1]
        objects['rvecAll']             = [rvec]
        objects['tvecAll']             = [tvec]
        objects['objectQ']             = [1]  # reliable
                
        return objects
 
    def init_pattern(self, pattern_size = (9,6), square_size = 10.5):
        # chessboard pattern init
        self.pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
        self.pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size
        
    def find_corners(self,color_image):
        image           = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        found, corners  = cv2.findChessboardCorners(image, tuple(self.pattern_size))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        if found:
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), term)
        return found, corners
    
    def draw_point_pairs(self, color_image, corners, point_ids):
        # plot a line
        clrs      = [(0, 0, 255),(0, 255, 0)]
        points_xy = np.zeros((2,2))
        for i,p in enumerate(point_ids):
            xy = corners[p].flatten().astype(np.int32)
            #color_image = cv2.drawMarker(color_image, xy, (255,0,0))
            color_image = cv2.circle(color_image, (xy[0],xy[1]), radius=10, color=clrs[i], thickness=2)
            points_xy[i,:] = xy

        dist = np.sqrt(np.sum((points_xy[0,:] - points_xy[1,:])**2))
        return color_image, dist  
    
    def draw_corners(self, color_image, corners):
        # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #cv2.drawChessboardCorners(color_image, self.pattern_size, corners, True)
        
        cv2.imshow('Calib Images',color_image)
        cv2.waitKey(1000)

        return color_image
    
    def get_object_pose(self, object_points, image_points, camera_matrix, dist_coeffs):
        #ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        # similarity to object
        ret, rvec, tvec, inliners = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        return rvec.flatten(), tvec.flatten()
    
    def calibrate_lens(self,image_list):
        "lens calibration "
        camera_matrix   = np.eye((3,3))
        dist_coeffs     = np.zeros(5)
        if len(image_list)<1:
            self.Print('Found no images','E')
            return camera_matrix, dist_coeffs
        
        img_points, obj_points = [], []
        h,w = 0, 0
        for imgName in image_list:
            # protect from non image files
            try:
                img         = cv2.imread(imgName)
            except BaseException as e:
                #print(e)
                self.Print('Skipping file %s' %imgName)
                continue
            
            if img is None: continue
            h, w            = img.shape[:2]
            found,corners = self.find_corners(img)
            if not found:
                raise Exception("chessboard calibrate_lens Failed to find corners in img")
            img_points.append(corners.reshape(-1, 2))
            obj_points.append(self.pattern_points)
            # show
            self.draw_corners(img, corners)
            
    #    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w,h))
        cv2.calibrateCamera(obj_points, img_points, (w,h), camera_matrix, dist_coeffs)
        # UD - similar
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w,h) ,cv2.CALIB_RATIONAL_MODEL,None)
        cv2.destroyAllWindows()
        
        return camera_matrix, dist_coeffs
 
    def Print(self, txt='',level='I'):
        
        if level == 'I':
            ptxt = 'I: CB: %s' % txt
            logging.info(ptxt)  
        if level == 'W':
            ptxt = 'W: CB: %s' % txt
            logging.warning(ptxt)  
        if level == 'E':
            ptxt = 'E: CB: %s' % txt
            logging.error(ptxt)  
           
        print(ptxt)

    def run_pose_estimation(self, fpath = ''):
        "compute calibration matrices"
        corner_list         = []
        obj_pose_list       = []
        img_list            = glob.glob(fpath)

        
        camera_matrix, dist_coeffs = self.calibrate_lens(img_list)

        for i, img_name in enumerate(img_list):

            # protect from non image files
            try:
                img         = cv2.imread(img_name)
            except BaseException as e:
                #print(e)
                self.Print('Skipping file %s' %img_name)
                continue
            
            if img is None: continue
            found, corners = self.find_corners(img)
            corner_list.append(corners)
            if not found:
                raise Exception("Failed to find corners in img # %d" % i)
            rvec, tvec = self.get_object_pose(self.pattern_points, corners, camera_matrix, dist_coeffs)
            #obj_pose_list.append(object_pose)    
    
    def compute_point_distances(self, fpath):
        "compute point distance ratio to understand distortion"
        img_list            = glob.glob(fpath)
        if len(img_list)<1:
            self.Print('Found no images','E')

        # which pairs are used to measure distances
        point_ids           = [[0,                      self.pattern_size[0]*(self.pattern_size[1]-1)],
                               [self.pattern_size[0]-1, self.pattern_size[0]*self.pattern_size[1]-1  ]]

        corner_list         = []

        for i, img_name in enumerate(img_list):

            # protect from non image files
            try:
                img         = cv2.imread(img_name)
            except BaseException as e:
                #print(e)
                self.Print('Skipping file %s' %img_name)
                continue
            
            if img is None: continue

            found, corners = self.find_corners(img)          
            if not found:
                raise Exception("Failed to find corners in img # %d" % i)
            
            corner_list.append(corners)

            # show detected points of interest
            dist_list = []
            for pair_ids in point_ids:
                img , d = self.draw_point_pairs(img, corners, pair_ids)
                dist_list.append(d)

            # compute ratio
            r = dist_list[0]/dist_list[1]
            self.Print('%s - ratio : %f' %(img_name,r))

            # show
            self.draw_corners(img, corners)
            


        cv2.destroyAllWindows()
        

      
# -------------------------- 
if __name__ == '__main__':
    print(__doc__)

    objChess     = ObjectChess()
    filePath     = r'C:\Users\udubin\Documents\Projects\MonoDepth\data\chess\*.*'
    #objChess.run_pose_estimation(filePath) # ok
    objChess.compute_point_distances(filePath)



    