
'''

Dataset creation from image - creates multiple patches per ROI.
Each patch is marked 0 or 1 according to the folder with laser Off or On.

Output : 
    x_train, x_test - NxD, MxD datasets with patch points, normalized to -1/1
    y_train, y_test - Nx1, Mx1 labels of the patch points, according to 0-Off/1-On of the laser


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 
    See README.md


'''

import numpy as np
import cv2 as cv
import unittest
import os

#from skimage.util.shape import view_as_windows
#from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from common import log, RectSelector
sys.path.append(r'..\Safety\src')
from extract_images_from_ros1bag import read_bin_file

# --------------------------------
#%% Help Fun
def MeanIoU(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)

def extract_patches(image, patch_size):
    """Extracts patches from an image.

    Args:
        image: The input image as a NumPy array.
        patch_size: The desired size of the patches (tuple of width and height).

    Returns:
        A list of patches extracted from the image.
    """

    patches = []
    height, width = image.shape[:2]

    for y in range(0, height - patch_size[1] + 1, patch_size[1]):
        for x in range(0, width - patch_size[0] + 1, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)

    return patches

import numbers
from numpy.lib.stride_tricks import as_strided

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : ndarray, shape (M[, ...])
        Input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.

    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_windows
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])

    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

# --------------------------------
#%% Data source
class DataSource:

    def __init__(self):

        # params
        self.mode            = 'ii2'  # video recording mode 
        self.vtype           = 0      # video source type
        self.frame_size      = (480,640)
        self.roi             = None #[0,0,self.frame_size[1],self.frame_size[0]]
        self.patch_size      = (16, 16)  # h,w - patch to extract pixels from
        self.point_size      = (3,3)     # h,w - for mask

        self.video_src       = None   # video source
        self.file_list       = None   # contrains bin file list

        self.tprint('Source is defined')

    def init_video(self, video_type = 11):
        # create video for test
        #w,h                 = self.frame_size
        if video_type == 1:
            "RealSense"
            fmode           = 'iig' 
            self.video_src  = RealSense(mode=fmode)                   

        elif video_type == 11:
            "with pattern"
            fname           = r"C:\Users\udubin\Documents\Projects\Safety\data\laser_power\video_ii2_000.mp4"
            fmode           = 'ii2' 
            self.video_src  = cv.VideoCapture(fname)

        elif video_type == 21:
            "pattern on - off"
            fname           = r"C:\Users\udubin\Documents\Projects\Safety\data\laser_power\video_ii2_001_ponoff.mp4"
            fmode           = 'ii2'  
            self.video_src  = cv.VideoCapture(fname)

        elif video_type == 31:
            "pattern on - off from bag files"
            fname           = r"C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726733151866586685_1280x720_step_1280_8uc1.bin"           
            self.video_src  = fname
            fmode           = 'img'  

        elif video_type == 32:
            "pattern on - off from bag files"
            fname           = r"C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data\image_1726733188232460976_1280x720_step_1280_8uc1.bin"           
            self.video_src  = fname
            fmode           = 'img'

        elif video_type == 41:
            "bag directory - sun and both proj are on"
            fname           = r"C:\Data\Safety\AGV\12_in_motion_both_prj_hall_ceramic_tile_sun\device_0_sensor_0_Infrared_1_image_data"
            self.video_src  = fname
            fmode           = 'bag'    

        elif video_type == 42:
            "bag directory - sun and both proj are off"
            fname           = r"C:\Data\Safety\AGV\12_in_motion_no_prj_hall_ceramic_tile_sun\device_0_sensor_0_Infrared_1_image_data"
            self.video_src  = fname
            fmode           = 'bag'                        

        else:
            self.tprint(f'Video type {video_type} is not supported','E')
            raise ValueError

        # read one frame to init
        self.mode           = fmode  
        self.vtype          = video_type
        self.tprint('Work mode : %s, video type : %d' %(fmode,video_type))
        return True
      
    def init_roi(self, roi_type = 1):
        "specify the relevant region"
        h,w     = self.frame_size
        roi     = [0,0,w,h]
        h2,w2   = h >>1, w>>1
        if roi_type == 1:
            roi = [w2-100,h2-100,w2+100,h2+100] # xlu, ylu, xrb, yrb
        elif roi_type == 2:
            roi = [w2-200,h2-200,w2+200,h2+200] # xlu, ylu, xrb, yrb
        else:
            pass
            
        self.tprint('ROI position %s' %str(roi))

        self.roi = roi       
        return roi 

    def convert_frame_from_input(self, frame):
        "extract depth and gray level channels"

        if self.mode == 'rgb':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]  
        elif self.mode == 'iig':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]   
        elif self.mode == 'ii2':
            self.frame_left  = frame[:,:,0] 
            self.frame_right = frame[:,:,1]
            self.frame_gray  = frame[:,:,2]                              
        else:
            self.tprint('bad mode','E')
            raise ValueError('bad mode')

        return True
    
    def get_image(self, fname):
        "get an infrared image frame from bin file"
        
        fsize               = (1280,720)
        fbpp                = 8

        img_array           = read_bin_file(fname,fsize,fbpp)
        self.frame_left     = img_array 

        # use ROI
        if self.roi is not None:
            x1,y1,x2,y2     = self.roi
            img_array       = img_array[y1:y2,x1:x2]

        return True, img_array.astype(np.float32)    

    def get_bag_directory(self):
        "get an infrared image frame from bin directory"

        # check if initialized
        fpath               = self.video_src
        if self.file_list is None: 
            files               = os.listdir(fpath)
            file_extensions     = [".bin"]
            filtered_files      = [file for file in files if os.path.isfile(os.path.join(fpath, file)) and file.endswith(tuple(file_extensions))]
            self.file_list      = filtered_files
            self.frame_count    = 0

        file_num            = len(self.file_list)
        if file_num < 1 or self.frame_count >= file_num:
            self.tprint('No bin files are found','W')
            return False, []

        # Iterate over files and process them
        file_name           = self.file_list[self.frame_count]
        file_path           = os.path.join(fpath, file_name)
        ret, img_array      = self.get_image(file_path)

        # check the number of files
        self.frame_count    = (self.frame_count + 1) % file_num

        return True, img_array      

    def get_frame(self):
        "get a single frame from the stream"
        # as the video frame.
        ret, frame           = self.video_src.read()  
        if not ret:
            return ret, []
                
        # convert channels
        ret                  = self.convert_frame_from_input(frame)
 
        frame_out            = frame
        self.first_time      = False
        self.frame_count     = self.frame_count + 1
        return True, frame_out.astype(np.float32)   
    
    def get_data(self):
        "get all the data structures"

        if self.mode == 'img':
            # read image data
            ret, frame          = self.get_image()

        if self.mode == 'bag':
            # read bag image data
            ret, frame          = self.get_bag_directory()            

        else: 
            # read video data
            ret, frame          = self.get_frame()  

        if not ret:
            self.tprint(f'Data source is not found ','E')

        return ret, frame  
    
    def transform_image_to_patch(self, fmap, mask_value = 1):
        # create patches from a single image. mask is defined as a enter of the image
            
        h,w             = self.patch_size
        nR,nC,ndim      = fmap.shape
        if ndim < 1:
            self.Print('Map channel size problem','E')
            return None, None
        
        # check for dimensions
        patch_shape     = (h,w,ndim)
        totalPixelhNum  = np.prod(patch_shape)*nR*nC
        if totalPixelhNum > 1e4:
            step        = 3
            self.tprint('The patch is too big or the image size is too big. Requires more than 1e8 pixels.','W')
            self.tprint('Reducing by factor 4','W')
        else:
            # compute step from the level
            step        = 2    
                
        # extract patches from the fmap 
        patches_i   = view_as_windows(fmap, patch_shape, step = step).squeeze()
        patches_i   = patches_i.reshape(-1, h * w * ndim)#[::8]
        
        # create mask
        #hp,wp       = self.point_size
        #nR2,nC2     = int(nR/2),int(nC/2) # if odd scales down
        fmsk        = np.zeros((nR,nC,1), dtype = np.uint8) + mask_value
        #fmsk[nR2-hp:nR2+hp, nC2-wp:nC2+wp, :] = 1

        # transform to patch            
        h2          = int(h/2)
        w2          = int(w/2)
        mask_shape  = (h,w,1)
        
        # take patches with step 1
        patches_c    = view_as_windows(fmsk, mask_shape, step = step).squeeze()
        # center of the patch
        patches_img  = patches_c[:,:,h2,w2]
        #patches_shape = patches_img.shape[:2]
        patches_m    = patches_img.reshape(-1,1)#[::8]
        
        # debug
        #self.ShowFeatureMask(fmap, fmsk, name = str(ind))

        return patches_i, patches_m
    
    def create_multiple_image_mask_patches(self, fmap, mask_value = 0):
        "create filters from multiple arrays"
        patches_img     = None
        patches_msk     = None
        
        # 4D arrays
        if (len(fmap.shape) != 4):
            self.Print('Image array must be 4D','E')
            return patches_img, patches_msk
        
        imgNum      = fmap.shape[0]       
        nChannels   = fmap.shape[3]
        if nChannels < 1:
            self.Print('Map channel size problem','E')
            return patches_img, patches_msk
        
        h,w                     = self.patch_size       
        patch_shape             = (h, w, nChannels)
        
        # run over all the maps
        patches_img                 = np.zeros((0,np.prod(patch_shape)))
        patches_msk                 = np.zeros((0,1))
        for k in range(imgNum):
            patches_i, patches_m,   = self.transform_image_to_patch(fmap[k,:,:,:], mask_value)
            patches_img             = np.vstack((patches_img,patches_i))
            patches_msk             = np.vstack((patches_msk,patches_m))
                

        return patches_img, patches_msk

    def create_patches_from_images_in_directory(self, image_directory = '', image_num = 3):
        "loads multiple bin images from a specified directory"

        self.video_src      = image_directory

        fpath               = image_directory
        files               = os.listdir(fpath)
        file_extensions     = [".bin"]
        filtered_files      = [file for file in files if os.path.isfile(os.path.join(fpath, file)) and file.endswith(tuple(file_extensions))]
        self.file_list      = filtered_files
        self.frame_count    = 0

        file_num            = len(self.file_list)
        if file_num < 1:
            self.tprint('No bin files are found','W')
            return False, []
        
        image_dataset   = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.          
        # subset of the images
        ri                  = np.random.permutation(file_num)[:image_num]
        for k in range(len(ri)):

            # Iterate over files and process them
            file_name           = self.file_list[ri[k]]
            file_path           = os.path.join(fpath, file_name)
            ret, img_array      = self.get_image(file_path)
            if not ret:
                self.tprint(f'Problem with {file_name}', 'W')
                break

            image_dataset.append(img_array)

        # add 4 dim since Gray images only
        image_dataset = np.array(image_dataset)
        image_dataset = image_dataset[:,:,:,np.newaxis]
        return image_dataset

    def create_dataset_from_directories(self, dirpaths = [''], file_num = 4, roi = (100,100,132,132),  mask_value = 1):
        "create X = NxD and Y = Nx1 - data set from multiple files"
        self.roi        = roi
        pimgs,pmsks     = None,None
        for dirpath in dirpaths:
            img_dataset     = self.create_patches_from_images_in_directory(dirpath, file_num)
            pimg,pmsk       = self.create_multiple_image_mask_patches(img_dataset,mask_value)
            if pimgs is None:
                pimgs,pmsks     = pimg,pmsk
            else:
                pimgs,pmsks     = np.vstack((pimgs,pimg)), np.vstack((pmsks,pmsk))

        return pimgs, pmsks
    
    def create_dataset(self, dir_list = [''], file_num = 4, roi_list = [(100,100,132,132)],  mask_values = [1]):
        "create train and test datasets from multiple directories"

        # check that all are compatible
        dir_num         = len(dir_list)
        assert dir_num == len(mask_values), 'Number directories and their mask values must be the same'
        
        pimgs,pmsks     = None,None
        # run over multiple ROIs
        for roi in roi_list:
            self.tprint('Collecting data for ROI : %s' %str(roi))
            self.roi        = roi
            for k in range(dir_num):
                dirpath         = dir_list[k]
                mask_value      = mask_values[k]
                img_dataset     = self.create_patches_from_images_in_directory(dirpath, file_num)
                pimg,pmsk       = self.create_multiple_image_mask_patches(img_dataset, mask_value)
                if pimgs is None:
                    pimgs,pmsks     = pimg,pmsk
                else:
                    pimgs,pmsks     = np.vstack((pimgs,pimg)), np.vstack((pmsks,pmsk))

        # normalize the data to -1:1 and msks to 0:1
        pimgs               = pimgs/128 - 1
        pmsks               = pmsks
        self.tprint(f'Mak : minimal value : {pmsks.min()}, maximal value {pmsks.max()}')

        # split the data on 2 datasets
        data_num            = pimgs.shape[0]
        split_num           = int(data_num*0.8)
        ri                  = np.random.permutation(data_num)
        x_train, y_train    = pimgs[ri[:split_num],:], pmsks[ri[:split_num]]
        x_test, y_test      = pimgs[ri[split_num:],:], pmsks[ri[split_num:]]

        self.tprint(f'Training point number : {x_train.shape[0]}')
        self.tprint(f'Testing point number  : {x_test.shape[0]}')
        return x_train, y_train, x_test, y_test 

    def check_accuracy(self, pmsk, pmsk_pred):
        #IOU for each class is..
        # IOU = true_positive / (true_positive + false_positive + false_negative).
        
        #Using built in keras function

        num_classes         = 2
#        IOU_keras           = MeanIoU(num_classes=num_classes)  
#        IOU_keras.update_state(pmsk, pmsk_pred)
#        self.Print("Mean IoU = %f" %IOU_keras.result().numpy())        
        #To calculate I0U for each class...
#        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
#        print(values)

        mres = MeanIoU(pmsk, pmsk_pred)
        self.tprint("Mean IoU = %f" %mres)
        
        
        # class1_IoU = values[1,1]/(values[1,1] + values[1,2] + values[1,3] + values[1,4] + values[2,1]+ values[3,1]+ values[4,1])
        # class2_IoU = values[2,2]/(values[2,2] + values[2,1] + values[2,1] + values[2,3] + values[1,2]+ values[3,2]+ values[4,2])
        # class3_IoU = values[3,3]/(values[3,3] + values[3,1] + values[3,2] + values[3,4] + values[1,3]+ values[2,3]+ values[4,3])
        # class4_IoU = values[4,4]/(values[4,4] + values[4,1] + values[4,2] + values[4,3] + values[1,4]+ values[2,4]+ values[3,4])
        
        # print("IoU Class 1 =", class1_IoU)
        # print("IoU Class 2 =", class2_IoU)
        # print("IoU Class 3 =", class3_IoU)
        # print("IoU Class 4 =", class4_IoU) 
        
    def check_images(self, fmsk, y_pred):
        # recover an image from the label
  
        hp,wp        = self.patch_size
        h2, w2       = int(hp/2), int(wp/2)
        
        
        # reshape
        h,w         = fmsk.shape[0] - hp + 1, fmsk.shape[1] - wp + 1
        lbl         = y_pred.reshape(h, w, -1).squeeze()
        
        # put
        fmsk_pred     = np.zeros(fmsk.shape[:2])
        fmsk_pred[h2:-h2+1,w2:-w2+1] = lbl
        
        return fmsk_pred

    def train_model(self, fimg):
        # creates model at a single levels
        time_s                      = time()
        
        pimg, pmsk                  = self.CreateMultiplemageMaskPatches(fimg)
        
        mdl, pmsk_pred              = self.FilterXGB(pimg, pmsk)        
        self.model                  = mdl
        
        self.Print('Train time : %f sec' %(time() - time_s))
        
        # check simple
        self.CheckPixelError(pmsk, pmsk_pred)
        
        # check IoU
        self.CheckAccuracy(pmsk, pmsk_pred)

        return True
    
    def show_feature_mask(self, feat = None, mask = None, name = "Train"):
        # not debug mode - do nothing
        if not self.debugOn:
            return
        
        if feat is None:
            self.Print('Load feature data first', 'E')
            return
        
        # support previous structures
        if len(feat.shape) < 4:
            feat = feat[np.newaxis,:,:,:]
        if mask is not None:
            if len(mask.shape) < 3: # add color channel
                mask = mask[:,:,np.newaxis]
            if len(mask.shape) < 4:
                mask = mask[np.newaxis,:,:,:]
        
        fNum = feat.shape[0]
        if mask is None:
            self.Print('No mask shown', 'E')
            mask = np.zeros_like(feat)
#        else:
#            mask = mask[0,:,:,0].astype(feat.dtype)
            
        
        n_col = 1
        n_row = fNum
        #plt.figure(40+level)
        plt.figure('%d:%s-Feat+Mask'%(self.figNum,name))
        #plt.figure('%d:L%d-Feat+Mask' %(self.figNum,level), figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.5)
        for f in range(n_row):
            
            featTmp = np.mean(feat[f,:,:,:],axis=2).squeeze()
            featTmp = featTmp/featTmp.max()
            maskTmp = mask[f,:,:,0].astype(feat.dtype)
            maskTmp = maskTmp/(maskTmp.max()+1e-6) #*featTmp.max()
            
            #img   = np.stack([featTmp, featTmp, featTmp/2+maskTmp/2],axis=2)
            img   = np.stack([maskTmp, featTmp, featTmp],axis=2).astype(np.float32)
            #img   = img/img.max()  # for float must 0:1 range
            plt.subplot(n_row, n_col, f + 1)
            plt.imshow(img,cmap = 'jet')
            plt.title('L%s-F%d' %(name,f), size=10)
            plt.xticks(())
            plt.yticks(())
        
        plt.show()          
        
        return  

    def show_patches(self, tempData, name = ''):
        # debug - show template data
        #if not self.debugOn:
        #    return
        
        # failed to match
        if len(tempData) < 1:
            return
        
        # support previous structures
        if len(tempData.shape) < 4:
            tempData = tempData[:,:,:,np.newaxis]
        
        # tempData - LxNxNxD
        labelNum,nR,nC,dim = tempData.shape

        tempData        = tempData.astype(np.uint8)
        montage_size    = int(np.ceil(np.sqrt(labelNum)))
        images          = np.zeros((montage_size*nR,montage_size*nC,dim), dtype = tempData.dtype)
        for r in range(montage_size):
            for c in range(montage_size):
                ri , ci = r*nR, c*nC
                k  = r*montage_size + c
                if k < labelNum:
                    images[np.int32(ri):np.int32(ri+nR), np.int32(ci):np.int32(ci+nC),:] = tempData[k,:,:,:]
        # too small
        while images.shape[1] < 500:
            images       = cv.resize(images,(int(images.shape[1]/0.5),int(images.shape[0]/0.5)))
        
        # construct the montages for the images
        cv.imshow("Montage :  %s" %name, images)
        cv.waitKey()
        return   

    def show_data(self):
        "draw relevant image data"
        if self.frame_left is None or self.frame_right is None:
            self.tprint('No images found')
            return False
            
        # deal with black and white
        #img_show    = np.concatenate((self.frame_left, self.frame_right), axis = 1)
        img_show    = self.frame_color

        while img_show.shape[1] > 2000:
            img_show    = cv.resize(img_show, (img_show.shape[1]>>1,img_show.shape[0]>>1), interpolation=cv.INTER_LINEAR)

        cv.imshow('Image L-R', img_show)
        #self.tprint('show done')
        ch  = cv.waitKey(1)
        ret = ch != ord('q')
        return ret

    def finish(self):
        # Close down the video stream
        if self.vtype < 10:
            self.video_src.release()

    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt)     

    def test_prediction(self, fimg):
        # test model 
        time_s                      = time()
        
        pimg, pmsk                  = self.CreateMultiplemageMaskPatches(fimg)
        
        pmsk_pred                   = self.PredictXGB(pimg, pmsk)        
        
        self.Print('Test time : %f sec' %(time() - time_s))
        
        # check simple
        self.CheckPixelError(pmsk, pmsk_pred)
        
        # check IoU
        self.CheckAccuracy(pmsk, pmsk_pred)

        return True   

    def test_images_predictions(self, fimg):
        # test model 
        imNum,h,w,d                 = fimg.shape
        fmsk_pred                   = np.zeros((imNum,h,w,1))
        for k in range(imNum):
            self.Print('Image %d ---------' %k)
            time_s                  = time()        
            fi                      = fimg[k,:,:,:]
            fie                     = fi[np.newaxis,:,:,:]
            pimg, pmsk              = self.CreateMultiplemageMaskPatches(fie)        
            pmsk_pred               = self.PredictXGB(pimg, pmsk) 
            fm_pred                 = self.check_images(fi, pmsk_pred)
            fmsk_pred[k,:,:,:]      = fm_pred[:,:,np.newaxis]
            
            # check simple
            self.CheckPixelError(pmsk, pmsk_pred)
            
            # check IoU
            self.CheckAccuracy(pmsk, pmsk_pred)
            
            #self.ShowMaskComparison(mask_ref, mask_pred, level = 1
        
            self.Print('Test time : %f sec' %(time() - time_s))
            
        
        self.show_feature_mask(fimg, fmsk_pred, name = 'Test - Predicted')
        return True  

# --------------------------------        
#%% Tests
class TestDataSource(unittest.TestCase):

    def test_data_source_rs(self):
        "show image from camera"
        p       = DataSource()
        srcid   = 1
        ret     = p.init_video(srcid)
        while ret:
            ret     = p.get_data()
            ret     = p.show_data()
        p.finish()
        self.assertFalse(ret)
  
    def test_data_source_video(self):
        "show image from video file"
        p       = DataSource()
        srcid   = 11
        ret     = p.init_video(srcid)
        while ret:
            ret     = p.get_data()
            ret     = p.show_data() and ret
        p.finish()
        self.assertFalse(ret)

    def test_transform_image_to_patch(self):
        "transform image to multiple patch objects"
        p               = DataSource()
        p.patch_size    = (3, 3)
        A               = np.arange(9*9).reshape(9,9,1).astype(np.float32)
        pimg,pmsk       = p.transform_image_to_patch(A,3)
        self.assertTrue(pimg.shape[0] > 0)


    def test_create_multiple_image_mask_patches(self):
        "testing data generation from multiple patches"
        p               = DataSource()
        p.patch_size    = (3, 3)
        A               = np.arange(9*9*6).reshape(6,9,9,1).astype(np.float32)
        pimg,pmsk       = p.create_multiple_image_mask_patches(A,1)
        self.assertTrue(pimg.shape[0] > 0)
        self.assertTrue(pimg.shape[0] == pmsk.shape[0])

    def test_create_patches_from_images_in_directory(self):
        "testing data set creation from multiple files in the directory"
        p               = DataSource()
        p.roi           = (200,500,250,550)
        dirpath         = r'C:\Data\Safety\AGV\12_in_motion_no_prj_hall_ceramic_tile_sun\device_0_sensor_0_Infrared_1_image_data'
        img_dataset     = p.create_patches_from_images_in_directory(dirpath, 3)
        #self.assertTrue(len(img_dataset) > 0)
        self.assertTrue(img_dataset.shape[0] == 3)

    def test_show_patches(self):
        "testing data set show with multiple files"
        p               = DataSource()
        #p.roi           = (400,500,500,600)
        p.roi           = (400,500,432,532)
        file_num        = 16
        dirpath         = r'C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        #dirpath         = r'C:\Data\Safety\AGV\12_in_motion_both_prj_hall_ceramic_tile_sun\device_0_sensor_0_Infrared_1_image_data'
        img_dataset     = p.create_patches_from_images_in_directory(dirpath, file_num)
        p.show_patches(img_dataset)
        self.assertTrue(img_dataset.shape[0] == file_num)        

    def test_create_dataset_from_directories(self):
        "data set creationm from a single directory"
        p               = DataSource()
        roi             = (400,500,432,532)
        file_num        = 16
        dirpath         = r'C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        pimg,pmsk       = p.create_dataset_from_directories([dirpath],file_num,roi, 1)
        self.assertTrue(pimg.shape[0] == pmsk.shape[0])
        print(pimg)

    def test_create_dataset(self):
        "data set creationm from a single directory"
        p               = DataSource()
        file_num        = 16

        dirpath1         = r'C:\Data\Safety\AGV\12_in_motion_no_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        dirpath2         = r'C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        dir_paths        = [dirpath1, dirpath2]
        mask_values      = [0,1]
        rois             = [(400,500,432,532), (800,500,832,532)]

        xt,yt,xv,yv       = p.create_dataset(dir_paths, file_num, rois, mask_values)
        self.assertTrue(xt.shape[0] == yt.shape[0])
        self.assertTrue(xv.shape[0] == yv.shape[0])
               



# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestDataSource("test_data_source_rs")) # ok
    #suite.addTest(TestDataSource("test_data_source_video")) # ok
    #suite.addTest(TestDataSource("test_transform_image_to_patch")) # ok
    #suite.addTest(TestDataSource("test_create_multiple_image_mask_patches")) # ok
    #suite.addTest(TestDataSource("test_create_patches_from_images_in_directory")) # ok
    #suite.addTest(TestDataSource("test_show_patches")) # ok
    #suite.addTest(TestDataSource("test_create_dataset_from_directories")) # ok
    suite.addTest(TestDataSource("test_create_dataset"))
    
    

    runner = unittest.TextTestRunner()
    runner.run(suite)

#%%
if __name__ == '__main__':
    #print (__doc__)
    RunTest()
    

   


    
 

