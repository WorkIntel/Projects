
"""
Created on Sep 04 16:53:13 2019

Analyze depth from optical flow and sensor angle rotation

-----------------------------
 Ver    Date     Who    Descr
-----------------------------
0102   31.09.24 UD     Robot chess.
0101   21.07.24 UD     Started chess. 
-----------------------------
"""



import numpy as np

import logging
import glob
import unittest
from scipy.spatial.transform import Rotation as Rot

import cv2 as cv




#%% Generate image pairs and points
class DataGenerator:
    "class to create images and correspondance points"
    def __init__(self):

        self.frame_size = (640,480)
        self.img        = None
   

    def add_noise(self, img_gray, noise_percentage = 0.01):
        "salt and pepper noise"
        if noise_percentage < 0.001:
            return img_gray


        # Get the image size (number of pixels in the image).
        img_size = img_gray.size

        # Set the percentage of pixels that should contain noise
        #noise_percentage = 0.1  # Setting to 10%

        # Determine the size of the noise based on the noise precentage
        noise_size = int(noise_percentage*img_size)

        # Randomly select indices for adding noise.
        random_indices = np.random.choice(img_size, noise_size)

        # Create a copy of the original image that serves as a template for the noised image.
        img_noised = img_gray.copy()

        # Create a noise list with random placements of min and max values of the image pixels.
        #noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)
        noise = np.random.choice([-10, 10], noise_size)

        # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
        img_noised.flat[random_indices] += noise
        
        self.tprint('adding image noise')
        return img_noised

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


        elif img_type == 8: # corner

            x,y             = np.meshgrid(np.arange(w),np.arange(h))   
            self.img        = np.ones((h,w))*250
            img_bool        = np.logical_and((x - w/2) < 0, (y - h/2) < 0)
            self.img[img_bool] = 230 # quarter                            

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

        self.img = self.add_noise(self.img, 0)
        self.frame_size = self.img.shape[:2]      
        return self.img

#%% Estimation
class DepthFromAngle:
    """ 
    Uses several images to match points and estimate the depth
    """

    def __init__(self):

        self.frame_size     = (640,480)
        self.focal_length   = 3 # mm
        self.pixel_size     = 1 #7.5e-3 # mm - includes decimation
        self.base_line      = 3 # mm - shift in focus due to rotation

        self.init_pattern()
        self.init_camera_params()

    def init_camera_params(self):
        """
        
        """
        self.cam_matrix = np.array([[1000,0,self.frame_size[0]/2],[0,1000,self.frame_size[1]/2],[0,0,1]], dtype = np.float32)
        self.cam_distort= np.array([0,0,0,0,0],dtype = np.float32)  

        isOk = self.load_params()
        if not isOk:
            self.Print('camera calibration is needed')  

    def init_pattern(self, pattern_size = (9,6), square_size = 10.0):
        # chessboard pattern init
        self.pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
        self.pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
        self.pattern_points *= square_size
        self.pattern_size    = pattern_size
        self.square_size     = square_size

    def save_params(self):
        "camera params save"
        # Save
        dictionary = {
            'cam_matrix' : self.cam_matrix,
            'cam_distort': self.cam_distort,
            }
        np.save('settings.npy', dictionary) 
        self.Print('saved to file')

    def load_params(self):
        # Load
        try:
            read_dictionary = np.load('settings.npy',allow_pickle='TRUE').item()
        except Exception as e:
            print(e)
            self.Print('File not found - run calibration')
            return False
        
        print(read_dictionary['cam_matrix']) # displays "world"   
        print(read_dictionary['cam_distort']) 

        self.cam_matrix    = read_dictionary['cam_matrix'] 
        self.cam_distort   = read_dictionary['cam_distort']
        return True
            
    def find_corners(self,color_image):
        image           = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        found, corners  = cv.findChessboardCorners(image, tuple(self.pattern_size))
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        if found:
            cv.cornerSubPix(image, corners, (11, 11), (-1, -1), term)
        return found, corners
    
    def draw_point_pairs(self, color_image, corners, point_ids):
        # plot a line
        clrs      = [(0, 0, 255),(0, 255, 0)]
        points_xy = np.zeros((2,2))
        for i,p in enumerate(point_ids):
            xy = corners[p].flatten().astype(np.int32)
            #color_image = cv.drawMarker(color_image, xy, (255,0,0))
            color_image = cv.circle(color_image, (xy[0],xy[1]), radius=10, color=clrs[i], thickness=2)
            points_xy[i,:] = xy

        dist = np.sqrt(np.sum((points_xy[0,:] - points_xy[1,:])**2))
        return color_image, dist  
    
    def draw_corners(self, color_image, corners, name = 'Calib Images'):
        # color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(color_image, self.pattern_size, corners, True)
        
        cv.imshow(name,color_image)
        cv.waitKey(1000)

        return color_image
    
    def convert_rvec_to_euler(self, rvec):
        "euler representation"
        #R           = Rot.from_quat(rvec).as_matrix()
        avec         = Rot.from_rotvec(rvec).as_euler('xyz',degrees=True)
        #avec        = Rot.from_matrix(R).as_euler('zyx',degrees=True)
        return avec
    
    def get_object_pose(self, object_points, image_points, camera_matrix, dist_coeffs):
        #ret, rvec, tvec = cv.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        # similarity to object
        ret, rvec, tvec, inliners = cv.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_EPNP)

        return rvec.flatten(), tvec.flatten()
    
    def calibrate_lens(self, image_list):
        "lens calibration "
        camera_matrix   = np.eye(3)
        dist_coeffs     = np.zeros(5)
        if len(image_list)<1:
            self.Print('Found no images','E')
            return camera_matrix, dist_coeffs
        
        img_points, obj_points = [], []
        h,w = 0, 0
        for imgName in image_list:
            # protect from non image files
            try:
                img         = cv.imread(imgName)
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
            
        #rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w,h))
        #cv.calibrateCamera(obj_points, img_points, (w,h), camera_matrix, dist_coeffs)
        # UD - similar
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w,h) ,cv.CALIB_RATIONAL_MODEL,None)
        
        
        print('RMS: ', rms)
        print(camera_matrix, dist_coeffs)
        #cv.destroyAllWindows()

        self.cam_matrix, self.cam_distort = camera_matrix, dist_coeffs
        self.save_params()  
        return True
    
    def compute_corner_distances(self, corners_left, corners_right):
        "distances between images"
        corners_left = corners_left.squeeze()
        corners_right = corners_right.squeeze()
        dx            = corners_left[:,0] - corners_right[:,0]
        dy            = corners_left[:,1] - corners_right[:,1]
        dist          = np.sqrt(dx**2 + dy**2)
        return dist.mean()

    def compute_point_distances(self, corners_left, corners_right):
        "distances between points scaled "
        corners_left = corners_left.squeeze()
        corners_right = corners_right.squeeze()
        dx            = corners_left[:,0] - corners_right[:,0]
        dy            = corners_left[:,1] - corners_right[:,1]
        dist_pix      = np.sqrt(dx**2 + dy**2)

        f             = self.focal_length    # mm
        pixel_size    = self.pixel_size    # = 10e-3 # mm - includes decimation
        b             = self.base_line     # = 1 # mm - shift in focus due to rotation
        alpha         = 5/180*np.pi # mm
  
        Z             = b*f/(dist_pix*pixel_size - 2*alpha*f)
        return Z.min(), Z.mean(), Z.max()
    
    def compute_undistorted_point_distances(self, corners_left, corners_right):
        "distances between undistorted points  "
        corners_left  = corners_left.squeeze()
        corners_right = corners_right.squeeze()
        dx            = corners_left[:,0] - corners_right[:,0]
        dy            = corners_left[:,1] - corners_right[:,1]
        dist_pix      = np.sqrt(dx**2 + dy**2)
        dist_pix      = dx
        
        pixel_size    = self.pixel_size    # = 10e-3 # mm - includes decimation
        b             = self.base_line     # = 1 # mm - shift in focus due to rotation
        alpha         = 5/180*np.pi # mm
  
        dist_rad      = dist_pix*pixel_size
        Z             = b/(2*alpha + dist_rad)
        return Z.min(), Z.mean(), Z.max()   

    def distance_between_images(self, fpath):
        "split data set on 3 and estimate distance between points "
        image_list  = glob.glob(fpath)
        
        img_points, obj_points = [], []
        h,w = 0, 0
        count = 0
        for imgName in image_list:
            # protect from non image files
            try:
                img         = cv.imread(imgName)
            except BaseException as e:
                #print(e)
                self.Print('Skipping file %s' %imgName)
                continue
            
            if img is None: continue
            h, w            = img.shape[:2]
            found,corners   = self.find_corners(img)
            if not found:
                raise Exception("chessboard calibrate_lens Failed to find corners in img")
            

            #corners_ud = cv.undistortPoints(corners, self.cam_matrix, self.cam_distort)

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(self.pattern_points)

            rvec, tvec = self.get_object_pose(self.pattern_points, corners, self.cam_matrix, self.cam_distort)
            avec       = self.convert_rvec_to_euler(rvec)
            self.Print('%d : T = %s, A = %s' %(count, str(tvec), str(avec)))

            # show
            self.draw_corners(img, corners,'Img %d' %count)
            count = (count + 1) % 3  

            # estimate distance
            if count == 0:
                corners_left  = img_points[-2] 
                corners_right = img_points[-1]   
                average_dist  = self.compute_corner_distances(corners_left, corners_right)  
                self.Print('%d : D = %s' %(count, str(average_dist)))   

        return True    
    
    def distance_between_points(self, fpath):
        "split data set on 3 and estimate distance between points "
        image_list  = glob.glob(fpath)
        
        img_points, obj_points = [], []
        tvecs = []
        h,w = 0, 0
        count = 0
        for imgName in image_list:
            # protect from non image files
            try:
                img         = cv.imread(imgName)
            except BaseException as e:
                #print(e)
                self.Print('Skipping file %s' %imgName)
                continue
            
            if img is None: continue
            h, w            = img.shape[:2]
            found,corners   = self.find_corners(img)
            if not found:
                raise Exception("chessboard calibrate_lens Failed to find corners in img")
            

            corners_ud = cv.undistortPoints(corners, self.cam_matrix, self.cam_distort)

            img_points.append(corners_ud.reshape(-1, 2))
            obj_points.append(self.pattern_points)

            rvec, tvec = self.get_object_pose(self.pattern_points, corners, self.cam_matrix, self.cam_distort)
            avec       = self.convert_rvec_to_euler(rvec)
            self.Print('%d : T = %s, A = %s' %(count, str(tvec), str(avec)))
            tvecs.append(tvec)

            # show
            self.draw_corners(img, corners,'Img %d' %count)
            count = (count + 1) % 3  

            # estimate distance
            if count == 0:
                tvec            = tvecs[-3]
                corners_left    = img_points[-2] 
                corners_right   = img_points[-1]   
                #z_min,z_mean,z_max  = self.compute_point_distances(corners_left, corners_right) 
                z_min,z_mean,z_max  = self.compute_undistorted_point_distances(corners_left, corners_right) 
                
                self.Print('%d : D = %s, Z = %s' %(count, str(tvec[2]), str([z_min,z_mean,z_max])))   

        return True        
 
    def Print(self, txt='',level='I'):
        
        if level == 'I':
            ptxt = 'I: DFA: %s' % txt
            logging.info(ptxt)  
        if level == 'W':
            ptxt = 'W: DFA: %s' % txt
            logging.warning(ptxt)  
        if level == 'E':
            ptxt = 'E: DFA: %s' % txt
            logging.error(ptxt)  
           
        print(ptxt)
   
    def compute_point_distances_old(self, fpath):
        "compute point distance ratio to understand distortion"
        img_list            = glob.glob(fpath)
        if len(img_list)<1:
            self.Print('Found no images','E')

        # which pairs are used to measure distances
        point_ids           = [[0,                      self.pattern_size[0]*(self.pattern_size[1]-1)],
                               [self.pattern_size[0]-1, self.pattern_size[0]*self.pattern_size[1]-1  ]]
        point_ids           = [[0,                      self.pattern_size[0]*(self.pattern_size[1]-1)],
                               [self.pattern_size[0]-4, self.pattern_size[0]*self.pattern_size[1]-4  ]]
        corner_list         = []

        for i, img_name in enumerate(img_list):

            # protect from non image files
            try:
                img         = cv.imread(img_name)
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

        cv.destroyAllWindows()

    def compute_using_optical_flow(self):
        "optical flow based depth estimation"
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        while(1):
            ret, frame2 = cap.read()
            if not ret:
                print('No frames grabbed!')
                break

            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame2', bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png', frame2)
                cv.imwrite('opticalhsv.png', bgr)
            prvs = next

        cv.destroyAllWindows()        
        


# ----------------------
#%% Tests
class TestDepthFromAngle(unittest.TestCase):

    def test_calibrate_lens(self):
        "lens distortion"
        p       = DepthFromAngle()
        f       = r'C:\Data\Depth\RobotAngle\*.png'
        imlist  = glob.glob(f)
        p.calibrate_lens(imlist)
        self.assertFalse(p.cam_matrix is None)  

    def test_distance_between_images(self):
        p       = DepthFromAngle()
        #f       = r'C:\Data\Depth\RobotAngle\*.png'
        f       = r'.\data\robot_cam_rotation\*.png'
        isOk    = p.distance_between_images(f)
        self.assertTrue(isOk)  

    def test_distance_between_points(self):
        p       = DepthFromAngle()
        f       = r'.\data\robot_cam_rotation\*.png'
        isOk    = p.distance_between_points(f)
        self.assertTrue(isOk)  


      
# -------------------------- 
if __name__ == '__main__':
    #print(__doc__)

     #unittest.main()
    suite = unittest.TestSuite()

    #suite.addTest(TestDepthFromAngle("test_calibrate_lens"))
    #suite.addTest(TestDepthFromAngle("test_distance_between_images")) # ok
    suite.addTest(TestDepthFromAngle("test_distance_between_points")) # ok

   
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # p0 = np.array([[364,616],[974,468],[878,305]])
    # pn = np.array([[378,613],[991,468],[894,304]])
    # pp = np.array([[326,624],[938,470],[843,304]])

    # def dist(p0,p1):
    #     d = np.sqrt((p0[:,0] - p1[:,0])**2 +(p0[:,1] - p1[:,1])**2)
    #     print(d)

    # dist(p0,pn)
    # dist(p0,pp)
                  


    