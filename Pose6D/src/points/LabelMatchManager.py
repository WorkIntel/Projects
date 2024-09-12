'''
Label Match Manager
Usage : 
    from LabelMatchManager import LabelMatchManager
    d = LabelMatchManager("string to you file")
    d.Test()
    
How to train:
   d = LabelMatchManager("string to you file")
   d.Test()
   
   
How to test:
   d = LabelMatchManager("string to you file")
   d.Test()

-----------------------------
 Ver	Date	 Who	Descr
-----------------------------
0101   28.08.24  UD     Created
-----------------------------

'''
   

import os
import numpy as np
import cv2 as cv
#import json
#import copy
#import random
import json

from glob import glob

from matplotlib import pyplot as plt
#from common import draw_str, RectSelector
#import video
#from skimage.feature import peak_local_max
from sklearn.cluster import KMeans

import time
import unittest
import logging



# ----------------
#%% Helpers
def rnd_warp(a):
    h, w    = a.shape[:2]
    T       = np.zeros((2, 3))
    coef    = 0.2
    ang     = (np.random.rand()-0.5)*coef
    scl     = 1+(np.random.rand()-0.5)*coef
    c, s    = np.cos(ang)*scl, np.sin(ang)*scl
    T[:2, :2] = [[c,-s], [s, c]]
    #T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv.warpAffine(a, T, (w, h), borderMode = cv.BORDER_REFLECT)

def cv_warp(img, x,y,a,s):
    center = (int(img.shape[1]//2+x), int(img.shape[0]//2+y))
    #center = (img.shape[1]//2, img.shape[0]//2)
    angle = a # degrees
    scale = s
    rot_mat = cv.getRotationMatrix2D( center, angle, scale)
    img_out = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT)
    return img_out

def cv_warp_array(img_gray, x, y, a, s):
    warpNum   = len(x)
    #img_array = np.tile(img_gray,[1,1,warpNum])
    img_array = np.repeat(img_gray[:, :, np.newaxis], warpNum, axis=2)
    for k in range(warpNum):
        img_out = cv_warp(img_gray,x[k],y[k],a[k],s[k])
        img_array[:,:,k] = img_out        
    return img_array

def cv_warp_img_points(img, points_xy, x,y,a,s):
    center       = (int(img.shape[1]//2+x), int(img.shape[0]//2+y))
    angle        = a # degrees
    scale        = s
    rot_mat      = cv.getRotationMatrix2D( center, angle, scale)
    img_out      = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags = cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT)
    #pointsXYw    = np.float32(points_xy).reshape(-1,1,2)
    #points_out   = cv.perspectiveTransform(pointsXYw, rot_mat)
    points_out   = points_xy @ rot_mat[:2,:2].T + rot_mat[:,2].T
    return img_out, points_out.squeeze()

def cv_warp_distance(img_warp_array1, img_warp_array2):
     # distance between arrays
     nR1,nC1,nT1 = img_warp_array1.shape
     nR2,nC2,nT2 = img_warp_array2.shape
     dist_mtrx   = np.zeros((nT1,nT2))
     for i1 in range(nT1):
         img1   = img_warp_array1[:,:,i1]
         bool1  = img1 > 0.1
         for i2 in range(nT2):
             img2   = img_warp_array2[:,:,i2]
             bool2  = img2 > 0.1
             
             valid  = bool1 & bool2
             dist   = np.mean(cv.absdiff(img1[valid] , img2[valid]))
             dist_mtrx[i1,i2] = dist
             
     return dist_mtrx
             
def plot_img_and_img_transformed(img, img_tr, val = 0):
#    fig, axs = plt.subplots(ncols=2, figsize=(16, 4), sharex=True, sharey=True)
#    axs[0].imshow(img)
#    axs[1].imshow(img_tr)
#    for ax in axs:
#        ax.set_xlim(0, 64)
#        ax.axis('off')
#    fig.tight_layout()
#    plt.show()
    
    plt.figure(32)
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Im 1')
    plt.subplot(1,2,2),plt.imshow(img_tr)
    plt.title('Im 2 : %d' %val), #plt.xticks([]), plt.yticks([])
    plt.show()          

    plt.pause(1)
    

# ----------------
#%% Deals with multuple templates
class LabelMatchManager:
    
    def __init__(self, config = None):
               
        self.state              = False  # object is initialized
        self.cfg                = config
        self.debugOn            = True
        self.figNum             = 1      # control names of the figures
        
        # constants
        self.MAX_FILES_TO_LOAD  = 16  # size of the template for labels
        self.TEMPLATE_SIZE      = 16  # size of the template for labels
        self.MAX_FILTER_NUM     = 8   # number of filters after decomposition
        self.WARP_ANGLES        = [0] #, 10,-10, 20,-20]  # augment
        self.WARP_SCALES        = [1]                  # augment
        self.CORR_THRESHOLD     = 0.85 # used in corr detection
        self.MIN_MATCH_DISTANCE = 30**2  # distnce between predicted corr and actual
        self.HITS_THRESHOLD     = 3
        
        self.trainDirList       = []
        self.testDirList        = []
        
#        # temp varaible
#        self.jsonPath           = ''
#        self.imagePath          = ''
        
        # help variables per file
        self.labelData          = []
        self.labelNames         = []

        # results after data extraction
        self.frameList          = [] # image frames
        self.templateList       = [] # templates per frame
        self.offsetList         = [] # offsets

        # arrays - datasets
        #self.arrayData          = None
        #self.arrayMask          = None

        self.Print('L-Manager is created')        
        
    def LoadTrainingFiles(self, topDirPath = '', fileNumToLoad = 0):
        # load jpg files from training directories
        ret = False
        if len(topDirPath) < 2:
            self.Print('Please specify training directory','E')
            return ret
        
        if not os.path.isdir(topDirPath):
            self.Print('Please specify valid path for the training directory : %s' %topDirPath,'E')
            return ret
            
        searchString = topDirPath  + '\**\*.json'
        filesList     = glob(searchString, recursive=True)
        
        fileNum       = len(filesList)
        if fileNum < 1:
            self.Print('Cannot find jpg files in directory or subdirectories of %s' %topDirPath,'E')
            return ret
        
        if fileNumToLoad < 1:
            fileNumToLoad = self.MAX_FILES_TO_LOAD
        
        # load only a small number
        fileNumToLoad   = np.minimum(fileNumToLoad,fileNum)
        #imgFiles = imgFiles[0:fileNum]
        #indx            = np.random.randint(0, fileNum-1, fileNumToLoad).ravel()
        #filesList       = random.sample(filesList, fileNumToLoad)
        filesList       = filesList[:fileNumToLoad]
        
        
        self.trainDirList           = filesList
        self.Print('Found %d training files. Loading %d files' %(fileNum,fileNumToLoad))
        return True
    
    def GetImageAndJsonFileNames(self, dataPath = None):
        # output
        jsonPathE               = ''
        framePathE              = ''
        
        if dataPath is None:
            return framePathE,  jsonPathE
        # print(dataPath)
        # derive from the frame path
        dataPathE, ext = dataPath.split('.')
        jsonPathE  = dataPathE + '.json'
        framePathE = dataPathE + '.jpg'
        #framePathE = dataPathE + '.' + ext
            
#        self.Print('Reading json file name %s' %jsonPathE, 'I')
#        self.Print('Reading image file name %s' %framePathE, 'I')
        
        # deal with image
        if not os.path.isfile(framePathE):       
            self.Print('Image file is not found %s' %framePathE, 'E')
            return framePathE,  jsonPathE

        # deal with json
        if not os.path.isfile(jsonPathE):       
            self.Print('Json file is not found %s' %jsonPathE, 'E')
            return framePathE,  jsonPathE
        
        
        # save
        #dataFound               = True
        #self.jsonPath           = jsonPathE
        #self.imagePath          = framePathE
        #self.state              = dataFound
        return framePathE,  jsonPathE   
     
    def GetPointsAndIdsFromJsonFile(self, jsonPath = None):
        # output
        objects_label_data  = []
        objects_label_names = []
        #self.labelNames     = object_label_names
        
        # check path
#        dataFound = self.GetImageAndJsonFileNames(jsonPath)
#        if dataFound is False:
#            self.Print('Can not find JSON or Image data','E')
#            return objects_label_data, objects_label_names
        
        # read json
        try:
            with open(jsonPath, 'rb') as f:
                jsonData = json.load(f)
                
            vers = jsonData.get('version')
            if vers is None:
                self.Print('Bad json file - no version. File should contain labeling points. %s' % jsonPath,'E')
                return objects_label_data, objects_label_names               
        except Exception as e:
            self.Print(e,'E')
            self.Print('Can not read JSON file %s' % jsonPath,'E')                
            return objects_label_data, objects_label_names
        
        # draw labels on the image
        objectInfo  = jsonData.get('objectInfo')
        if len(objectInfo) < 1:
            self.Print('Bad JSON file %s - object number must be at least 1' % jsonPath,'W')                
            return objects_label_data, objects_label_names


        vers_num = int(vers[:2])
        # new version
        if vers_num <= 20:
            # check all files in the views
            obj_num     = len(objectInfo)

            for k  in range(obj_num):
                try:
                    pLabels = objectInfo[k]['labelNames']
                except:
                    pLabels = objectInfo[k]['labels']
                pXY         = objectInfo[k]['points']
                objId       = k + 1
                
                nLabels     = len(pLabels)
                nPoints     = len(pXY)
                obj_labeled = np.zeros((nPoints,3))
                label_names = pLabels
                
                if nLabels != nPoints:
                    self.Print('Bad JSON file %s - point and label missmatch in object %d' % objId,'W')                
                    break 
                    
                for ii in range(nLabels):
                    obj_labeled[ii,0] = pXY[ii][0]
                    obj_labeled[ii,1] = pXY[ii][1]
                    obj_labeled[ii,2] = int(objId)-1
                    
                objects_label_data.append(obj_labeled)
                objects_label_names.append(label_names)
                    
                    
            
        elif 20 < vers_num and vers_num < 23:
            self.Print('Old JSON version','W')
            
            # new version
            obj_num     = len(jsonData['objectInfo'])
            point_num   = len(jsonData['shapes'])
            obj_labeled = np.zeros((point_num,3))
            label_names = []
            
            for k in range(point_num):

                pLabels = jsonData['shapes'][k]['label']
                pXY     = jsonData['shapes'][k]['points']
                objId   = jsonData['shapes'][k]['group_id']
                
                obj_labeled[k,0] = pXY[0][0]
                obj_labeled[k,1] = pXY[0][1]
                obj_labeled[k,2] = int(objId)-1
                
                label_names.append(pLabels.reshape((-1,)))
        else:
            
            obj_num     = len(jsonData['objectInfo'])
            if obj_num < 1:
                self.Print('No objects found in %s' %jsonPath,'E')
                return objects_label_data, objects_label_names  
            
            point_num   = jsonData['objectInfo'][0].get('pointNum')
            if point_num < 3:
                self.Print('No points found in %s' %jsonPath,'E')
                return objects_label_data, objects_label_names  
             
            # in the version 2308 - numbering could be upto 50 objects - non consequitive
            if not ('projectedShapes' in jsonData):
                self.Print('The labeled data is not exported correctly','E')
                return objects_label_data, objects_label_names
            
            for i in range(obj_num):
                projShape   = jsonData['projectedShapes'][i]
                
                # print(projShape)
                obj_labeled = np.zeros((point_num,4))
                label_names = []
                for k in range(len(projShape)):
                    shape   = projShape[k]
                    objType = shape.get("object_type", "0")
#                    if self.objectType != objType:
#                        continue
                    pLabels = shape['label']
                    pXY     = shape['points'][0]
                    objId   = shape['group_id']
                    
                    
                    obj_labeled[k,0] = pXY[0]
                    obj_labeled[k,1] = pXY[1]
                    obj_labeled[k,2] = int(objId)-1
                    # obj_labeled[k,3] = int(objType) 
                    label_names.append(pLabels)
                    
                objects_label_data.append(obj_labeled)
                objects_label_names.append(label_names)
            
        #self.labelNames = objects_label_names
        
        return objects_label_data , objects_label_names
    
    def GetImageDataFromFile(self, framePath = None):
        # output
        frame = np.zeros(0)
        
#        # check path
#        dataFound = self.GetImageAndJsonFileNames(framePath)
#        if dataFound is False:
#            self.Print('Can not find JSON or Image data','E')
#            return ret
        
        if framePath is None:
            framePath = self.imagePath
            
        # deal with image
        if not os.path.isfile(framePath):       
            self.Print('Image file is not found %s' %framePath, 'E')
            return frame
        
        try:
            frame = cv.imread(framePath)
        except Exception as e:
            self.Print(e,'E')
            self.Print('Can not read image file %s' % framePath,'E')                
            return frame
        
        # save
        frameH, frameW      = frame.shape[:2]
        
        self.imageData      = frame
        #self.Print('Reading file %s image of size (%dx%d)' %(framePath,frameW,frameH))
        
        return frame
    
    def GetLabelDataFromFile(self, jsonPath = None, objId = 0):
        # output
        labelData, labelNames = [], []
    
        # deal with json
        objects_data, label_data = self.GetPointsAndIdsFromJsonFile(jsonPath)
        #objects_data = [arr for arr in objects_data if not np.all((arr == 0))]
        objects_num  = len(objects_data)
        if objects_num < 1:       
            self.Print('Can not find object data %s' %jsonPath, 'E')
            return labelData, labelNames
        
        if not(objects_num > objId):       
            self.Print('Requested objects Id %d exceeds number of objects in the file %s' %(objId,jsonPath), 'E')
            return labelData, labelNames
        
        
        labelData          = objects_data[objId]
        labelNames         = label_data[objId]
        
        point_num, frameW  = labelData.shape[:2]
        
        self.objectId     = objId
        #self.Print('Reading file %s, object Id %d with %d points' %(jsonPath,objId,point_num))
        
        return labelData, labelNames

    def ComputeImageTempates(self, img = None, labels = None):
        # find closest feature points to each human label
        tempData, tempOffsets         = [], []
        
        if self.TEMPLATE_SIZE < 5:
            self.Print('Template side is too small','E')
            return tempData, tempOffsets
        
#        if not self.state:
#            self.Print('Requested objects view is not intialized','E')
#            return tempData, tempOffsets
        
        if img is None:
            self.Print('Image data is not intialized','E')
            return tempData, tempOffsets
            
        if labels is None:
            self.Print('Label data is not intialized','E')
            return tempData, tempOffsets
           
        labelNum    = len(labels)
        if labelNum < 4:
            self.Print('Label data is not correct','E')
            return tempData, tempOffsets
        
        # extract xy - offset
        labelXY     = np.zeros((2,labelNum))
        for k in range(labelNum):
            labelXY[0,k]     = labels[k][0] # 0 is the index number
            labelXY[1,k]     = labels[k][1]     
            
        # there joints that are nagative - need protection
        dim         = img.shape[2]
        width       = img.shape[1]
        height      = img.shape[0]
        
        templateSide     = self.TEMPLATE_SIZE
        templateSideHalf = np.int32(templateSide/2)
        
        tempData     = np.zeros((templateSide,templateSide,dim,labelNum),dtype=img.dtype)
        tempOffsets  = np.zeros((2,labelNum),dtype= np.int32)

        for k in range(labelNum):
            c_x     = np.int32(labels[k][0]) # 0 is the index number
            c_y     = np.int32(labels[k][1])
            
            # correct if hits boundaries
            offset_x = min(max((templateSideHalf - c_x), 0), (width-1-templateSideHalf - c_x))
            offset_y = min(max((templateSideHalf - c_y), 0), (height-1-templateSideHalf - c_y))
            
#            ind_x     = np.arange(c_x + offset_x - templateSideHalf,c_x + offset_x + templateSideHalf - 1)
#            ind_y     = np.arange(c_y + offset_y - templateSideHalf,c_y + offset_y + templateSideHalf - 1)
            
            #tempData[:,:,:,k] = img[ind_y,ind_x,:]
            tempData[:,:,:,k] = img[c_y + offset_y - templateSideHalf: c_y + offset_y + templateSideHalf ,c_x + offset_x - templateSideHalf: c_x + offset_x + templateSideHalf,:]
            tempOffsets[0,k] = c_x + offset_x
            tempOffsets[1,k] = c_y + offset_y
            
        #self.Print('Templates are extracted')  

        return tempData, tempOffsets
            
    def ConvertFilesToObjects(self):
        # Extract the templates  and offsets
        self.frameList , self.templateList , self.offsetList  = [], [], []

        fileNum     = len(self.trainDirList)
        if fileNum < 1:
            self.Print('Cannot find jpg files.','E')
            return False
        
        objectId        = 0
        for k in range(fileNum):
            # 
            jsonPath            = self.trainDirList[k]
            imgPath, jsonPath   = self.GetImageAndJsonFileNames(jsonPath)
            if len(imgPath) < 1:
                self.Print('No image data for json file %s' %jsonPath,'E')
                continue 
                
            label_data, label_names  = self.GetLabelDataFromFile(jsonPath, objectId)
            if len(label_data) < 1:
                self.Print('No label data in json file %s' %jsonPath,'E')
                continue 
            
            frame_data             = self.GetImageDataFromFile(imgPath)
            if len(frame_data) < 1:
                self.Print('Problem reading image file %s' %imgPath,'E')
                continue 
            
            # extract tempaltes
            tempData, tempOffsets = self.ComputeImageTempates(frame_data, label_data)
            if len(tempData) < 1:
                self.Print('Problem converting to templates','E')
                continue
            
            # save
            self.frameList.append(frame_data)
            self.templateList.append(tempData)
            self.offsetList.append(tempOffsets)
            
            # show
            #self.ShowImageLabelData(frame_data, tempOffsets, label_data, 'File - %d' %k)
            
        objNum          = len(self.templateList)
        
        #self.objectList = objectDataList
        self.labelNames = label_names
        self.Print('Found %d objects' %objNum)
        return True
    
    def ComputeImageAugmentation(self, img, offsetArray):
        # Use lie derivatives idea
        hp, wp, dp      = img.shape
        
        # define shifts
        max_angle, max_scale       = 15, 0.2
        num_angle, num_scale       = 3, 3
        
        rng_angle       = np.linspace(-max_angle,  max_angle, num_angle) 
        rng_scale       = np.linspace(1-max_scale, 1+max_scale, num_scale) 
        
        rng_angle       = np.array(self.WARP_ANGLES) 
        rng_scale       = np.array(self.WARP_SCALES)

        
        a, s            = np.meshgrid(rng_angle, rng_scale)  
        a, s            = a.ravel(), s.ravel() 
        n_warps         = len(a)
        
        # extract xy - offset
        points_xy       = offsetArray[:,:2]
        n_points        = points_xy.shape[0]


        # run over warps
        imgArray        = np.zeros((hp, wp, dp, n_warps))
        pointArray      = np.zeros((n_points, 2, n_warps))
        for ii in range(n_warps):
            #img_gray1           = templateArray[:,:,0,i1]
            img_w, points_xy_w  = cv_warp_img_points(img, points_xy, 0,0,a[ii],s[ii])
            
            imgArray[:,:,:,ii]  = img_w
            pointArray[:,:,ii]  = points_xy_w
                
            #debug
            #plot_img_and_img_transformed(img, img_w, 0)
            
            # show
            #self.ShowImageLabelData(img_w, points_xy_w, self.labelNames, 'Aug - %d' %ii)
                                   
        return imgArray, pointArray  
    
    def ConvertFilesToObjectsWithAugmentation(self):
        # Extract the templates  and offsets
        self.frameList , self.templateList , self.offsetList  = [], [], []

        fileNum     = len(self.trainDirList)
        if fileNum < 1:
            self.Print('Cannot find jpg files.','E')
            return False
        
        objectId        = 0
        for k in range(fileNum):
            # 
            jsonPath            = self.trainDirList[k]
            imgPath, jsonPath   = self.GetImageAndJsonFileNames(jsonPath)
            if len(imgPath) < 1:
                self.Print('No image data for json file %s' %jsonPath,'E')
                continue 
                
            label_data, label_names  = self.GetLabelDataFromFile(jsonPath, objectId)
            if len(label_data) < 1:
                self.Print('No label data in json file %s' %jsonPath,'E')
                continue 
            
            frame_data             = self.GetImageDataFromFile(imgPath)
            if len(frame_data) < 1:
                self.Print('Problem reading image file %s' %imgPath,'E')
                continue 
            
            # augmentation
            self.labelNames = label_names # for debug
            frame_data_aug, label_data_aug = self.ComputeImageAugmentation(frame_data, label_data)            
            
            # extract tempaltes
            for k in range(label_data_aug.shape[2]):
                
                frame_data_k, label_data_k = frame_data_aug[:,:,:,k], label_data_aug[:,:,k]
                tempData, tempOffsets  = self.ComputeImageTempates(frame_data_k, label_data_k)
                if len(tempData) < 1:
                    self.Print('Problem converting to templates','E')
                    continue
                
                # save
                self.frameList.append(frame_data)
                self.templateList.append(tempData)
                self.offsetList.append(tempOffsets)
            
            # show
            #self.ShowImageLabelData(frame_data, tempOffsets, label_data, 'File - %d' %k)
            
        objNum          = len(self.templateList)
        
        #self.objectList = objectDataList
        self.labelNames = label_names
        self.Print('Found %d objects' %objNum)
        return True    
    
    def GetTemplatesPerLabel(self,  labelId = 1):
        # show object data
        
        frameNum      = len(self.templateList)
        if frameNum < 1:
            self.Print('Load object training data','E')
            return None
        
        nR, nC, dim, labelNum      = self.templateList[0].shape
        if labelNum < 5:
            self.Print('Label data is not correct - must have at least 6-7 labels','E')
            return None
        
        # container
        tempData     = np.zeros((nR,nC,dim,frameNum),dtype=np.uint8)
        tempOffs     = np.zeros((2,frameNum))
        
        for k in range(frameNum): 
            tmpObjData          = self.templateList[k]
            tempData[:,:,:,k]   = tmpObjData[:,:,:,labelId]
            tmpOffsData         = self.offsetList[k]
            tempOffs[:,k]       = tmpOffsData[:,labelId]
            
        return tempData, tempOffs    
    
    def ComputeDecompositionSVD(self, templateArray):
        # output
        #img               = self.ImagePreprocessing(img)
        hp, wp, dp, n_labels  = templateArray.shape
            
        #dataPos              = imgPosArray.transpose(2,0,1).reshape(-1,data.shape[1])            
        dataPos              = templateArray.reshape(-1,n_labels)            
        
        # remove mean
        dataPos              = dataPos - np.mean(dataPos,axis = 0)
        
        # decompose
        up, sp, vhp          = np.linalg.svd(dataPos, full_matrices=False)
        
        # compute dominating values
        svec                = sp
        svec_prob           = np.cumsum(svec)
        svec_prob           = svec_prob/svec_prob[-1]
        n_filters           = np.argmax(svec_prob[svec_prob < 0.85])
        n_filters           = n_filters + 1 # if the first index 0
        
        # limit the nuber of filters -save compute
        n_filters           = np.minimum(n_filters, self.MAX_FILTER_NUM)
        
        # loss estimates
        totalEnergy         = np.sum(sp)
        coveredEnergy       = np.sum(sp[0:n_filters])
        self.Print('Covered energy %4.1f percent' %(coveredEnergy/totalEnergy*100))
        
        # filters
        flt                 = up[:,0:n_filters]
        #fs1_montage         = montage(fs1, rescale_intensity=True)
        
          
        # create image
        patch_shape          = (hp,wp,dp,n_filters)
        flt_image            = flt.reshape(patch_shape)
        
        return flt_image #.astype(np.uint8)
    
    def ComputeDecomposition(self, templateArray):
        "K-NN decomposition"
        # output
        #img               = self.ImagePreprocessing(img)
        hp, wp, dp, n_labels  = templateArray.shape
            
        #dataPos              = imgPosArray.transpose(2,0,1).reshape(-1,data.shape[1])            
        dataPos              = templateArray.reshape(-1,n_labels).T            
        
        # remove mean
        #dataPos              = dataPos - np.mean(dataPos,axis = 0)
        
        # decompose
        n_filters           = self.MAX_FILTER_NUM
        kmeans              = KMeans(n_clusters=n_filters, random_state=0, n_init="auto").fit(dataPos)
        
        # loss estimates
        # totalEnergy         = np.sum(sp)
        # coveredEnergy       = np.sum(sp[0:n_filters])
        # self.Print('Covered energy %4.1f percent' %(coveredEnergy/totalEnergy*100))
        
        # filters
        flt                 = kmeans.cluster_centers_.T       
          
        # create image
        patch_shape          = (hp,wp,dp,n_filters)
        flt_image            = flt.reshape(patch_shape)
        
        return flt_image #.astype(np.uint8)    
    
    def ComputeFilters(self, templateArray):
        # output
        #img               = self.ImagePreprocessing(img)
        hp, wp, dp, n_labels  = templateArray.shape
        if dp != 3:
            self.Print('Filters should have color','E')
            return
            
        warp_angles         = self.WARP_ANGLES
        warp_num            = len(warp_angles)
        
        filterArray         = np.zeros((hp,wp,warp_num,n_labels))
        for k in range(n_labels):
            templGray           = cv.cvtColor(templateArray[:,:,:,k],cv.COLOR_RGB2GRAY)
            for w in range(warp_num):
                ang                 = warp_angles[w]
                img_out             = cv_warp(templGray,0,0,ang,1)
                filterArray[:,:,w,k]= img_out            
        
        return filterArray #.astype(np.uint8)  
    
    def ComputeFiltersAugmented(self, templateArray):
        # create filter array from already augmented image data
        #img               = self.ImagePreprocessing(img)
        hp, wp, dp, n_frames  = templateArray.shape
        if dp != 3:
            self.Print('Filters should have color','E')
            return
     
        filterArray         = np.zeros((hp,wp,n_frames))
        for k in range(n_frames):
            templGray           = cv.cvtColor(templateArray[:,:,:,k],cv.COLOR_RGB2GRAY)
            filterArray[:,:,k]= templGray            
        
        return filterArray #.astype(np.uint8) 

    def ComputeCorrelationWarpedTemplates(self, imgArray, coeffArray, lblId = 0):
        # computes correlation  - per single label   
        start_time          = time.time()
        coeffH, coeffW,  coeffC, coeffNum = coeffArray.shape
            
        #searchH, searchW, searchNum = imgArray.shape
        searchH, searchW, searchC = imgArray.shape
        
        # checks
        if searchH < coeffH or searchW < coeffW :  # 
            self.Print('searchSide less than coeffSide ','E')
            return None
        
        # debug
        corrH     = searchH - coeffH  +  1
        corrW     = searchW - coeffW  +  1
        corrData  = np.zeros((corrH,corrW,coeffC,coeffNum),dtype=np.float32)
        imgArray  = imgArray.astype(np.float32)
        coeffArray = coeffArray.astype(np.float32)
        imageGray = cv.cvtColor(imgArray,cv.COLOR_RGB2GRAY)

        
        coeffW2   = coeffW/2
        coeffH2   = coeffH/2   
        
        peakOffset = np.zeros((2,coeffC,coeffNum))
        peakValue = np.zeros((1,coeffC,coeffNum))
        
        score     = 0.0
        for k in range(coeffNum):
            for c in range(coeffC):
            
                templGray   = coeffArray[:,:,c,k]
                result      = cv.matchTemplate(imageGray, templGray, cv.TM_CCOEFF_NORMED)                
                corrData[:,:,c,k] =  result
            
                # compute max score
                min_val, max_val, min_indx, max_indx = cv.minMaxLoc(result)
                score         += max_val
                peakOffset[0,c,k] = max_indx[0] + coeffW2
                peakOffset[1,c,k] = max_indx[1] + coeffH2
                peakValue[0,c,k]  = max_val
                
                self.Print('Filt %d : Ang %+2d, max: %f' %(k, self.WARP_ANGLES[c], max_val),'I')
                # mark in color
                #corrData[max_indx[1],max_indx[0],c,k] = 0
            
        # center
        #peakOffset = peakOffset + 1 # small correction 
        score      = peakValue.max()
        maxindex    = peakValue.argmax()
        peak_index = np.unravel_index(maxindex, peakValue.shape)
        peak_c,peak_k = peak_index[1], peak_index[2]
        
        
        #self.Print('C-Match done with score %f' %score)
        #self.Print('Index:')
        #print(peak_index)
        #self.Print('Offsets:')
        #print(peakOffset[:,peak_c,peak_k])
        
#        #debug
#        plt.figure(2)
#        plt.subplot(1,1,1),plt.imshow(corrData.squeeze(),cmap = 'gray')
#        plt.title('corrData'), #plt.xticks([]), plt.yticks([])
#        plt.show()  

        self.ShowFeatures(corrData, lblId)        
        
         # check
        elapsed_time = time.time() - start_time
        self.Print('Best %d : Ang %d, max: %f' %(peak_k, self.WARP_ANGLES[peak_c], score),'I')
        self.Print('Processing score %f in %f sec' %(score,elapsed_time))
        return score, peak_k, peak_c   

    def ComputeCorrelationAugmentedCoeff(self, imgArray, coeffArray, lblId = 0):
        # computes correlation  - per single label   
        #start_time          = time.time()
        coeffH, coeffW,  coeffNum, labelNum = coeffArray.shape
            
        #searchH, searchW, searchNum = imgArray.shape
        searchH, searchW, searchC = imgArray.shape
        
        # checks
        if searchH < coeffH or searchW < coeffW :  # 
            self.Print('searchSide less than coeffSide ','E')
            return None
        
        # debug
        corrH     = searchH - coeffH  +  1
        corrW     = searchW - coeffW  +  1
        #corrData  = np.zeros((corrH,corrW,coeffNum,labelNum),dtype=np.float32)
        #imgArray  = imgArray.astype(np.float32)
        coeffArray = coeffArray.astype(np.float32)
        imageGray = cv.cvtColor(imgArray,cv.COLOR_RGB2GRAY).astype(np.float32)

        
        coeffW2   = coeffW/2
        coeffH2   = coeffH/2   
        
#        peakOffset = np.zeros((2,labelNum,coeffNum))
#        peakValue = np.zeros((1,labelNum,coeffNum))
        peakXY    = []
        
        score     = 0.0
        for c in range(labelNum):
            start_time          = time.time()
            #corrDataLabel  = np.zeros((corrH,corrW,coeffNum),dtype=np.float32)
            peakXYF = []
            for k in range(coeffNum):
            
                templGray   = coeffArray[:,:,k,c]
                result      = cv.matchTemplate(imageGray, templGray, cv.TM_CCOEFF_NORMED)                
                #result      = cv.matchTemplate(imageGray, templGray, cv.TM_CCORR_NORMED)                
                #corrDataLabel[:,:,k] =  result
                
                yx                  = peak_local_max(result,min_distance=5,threshold_abs=self.CORR_THRESHOLD, exclude_border=True).astype(np.float32) # out <y,x> Peaks are the local maxima in a region of 2 * min_distance + 1 (i.e. peaks are separated by at least min_distance).
                if yx.shape[0] > 0 :
                    #xyk = np.concatenate([xy.astype(np.int32), np.full((xy.shape[0],1),k,dtype=np.int32)], axis=1)
                    
                    # switch x and y and add offsets 
                    xy          = yx[:,:2]
                    xy[:,0],xy[:,1] = xy[:,1]+coeffW2, xy[:,0]+coeffH2
                    xyk = np.concatenate([xy, np.full((yx.shape[0],1),k,dtype=yx.dtype)], axis=1)
                    peakXYF.append(xyk)
                    
                    # debug
                    self.ShowImageAndDataXY(imgArray,xyk[:,:2],'%d-%d'%(c,k))
            
#                # compute max score
#                min_val, max_val, min_indx, max_indx = cv.minMaxLoc(result)
#                #score         += max_val
#                peakOffset[0,c,k] = max_indx[0] + coeffW2
#                peakOffset[1,c,k] = max_indx[1] + coeffH2
#                peakValue[0,c,k]  = max_val
                
                #self.Print('Label : %d, Filt %d : max: %f' %(c, k,  max_val),'I')
                # mark in color
                #corrData[max_indx[1],max_indx[0],c,k] = 0
                
            
            # for the specific label do analysis
            #score      = corrDataLabel.max()
            #thr        = np.maximum(score*0.9, self.CORR_THRESHOLD)
            #corrDataLabel[corrDataLabel < thr] = 0
            
            # save
            #corrData[:,:,:,c] = corrDataLabel
            peakXY.append(peakXYF)
            # check
            elapsed_time = time.time() - start_time
            #self.Print('Best %d : Ang %d, max: %f' %(peak_k, self.WARP_ANGLES[peak_c], score),'I')
            self.Print('Label : %d, Processing score %f in %f sec' %(c, score,elapsed_time))
            
#        # debug
#        maxi       = np.argmax(peakValue, axis = 2)
#        dataXY     = peakOffset[:,range(labelNum),maxi].squeeze()
#        #self.ShowImageAndDataXY(imgArray,dataXY,'Label All')
#        self.ShowImageLabelData(imgArray, dataXY, list(range(labelNum)), 'Debug' )


        # like sigmoid squash
#        corrData   = 5*(corrData - thr)
#        corrData   = 1/(1 + np.exp(-corrData))
#        corrData[corrData < self.CORR_THRESHOLD] = 0
        

#        self.ShowFeatures(corrData, lblId)        
        
        return peakXY  
    
    def ComputePredictedLabelPositions(self, peakArray, offsetArray, lblIds):
        # predict point position using mutual info
        # peakArray is the list of n_labels with M records of peack
        #img               = self.ImagePreprocessing(img)
        start_time          = time.time()
        #hp, wp, n_frames, n_labels  = corrData.shape
        n_labels                    = len(peakArray)
        dp, m_frames, m_labels      = offsetArray.shape
            
#        # the frames were augmented by angles and scale
#        warp_angles         = self.WARP_ANGLES
#        warp_scales         = self.WARP_SCALES
#        warp_num            = len(warp_angles)*len(warp_scales)
#        
#        # will result in bug - need to understand how angles and scales has beeen created
#        assert len(warp_scales) < 2         
#        file_num_f          = n_labels/warp_num
#        file_num            = int(file_num_f)            
#        if (file_num_f - file_num) != 0:
#            self.Print('Augmentation problem','E')
#            return
#        
#        warp_angles_ind     = np.remainder(np.arange(n_frames), warp_num)
#        warp_angles_rad     = [np.deg2rad(warp_angles[k]) for k in warp_angles_ind]
        
        coeffWidth         = self.TEMPLATE_SIZE
        hi, wi              = self.frameList[0].shape[:2]
        hp, wp              = hi + coeffWidth - 1, wi + coeffWidth - 1
        
        # hit accumulator
        corrAccum           = np.zeros((hp, wp, n_labels))
        #inverseReference    = np.zeros((hp, wp, n_frames, n_labels))
        for mf in range(n_labels):
            
            peakXYF              = peakArray[mf]
            frameNumF            = len(peakXYF)
            corrAccumF           = corrAccum[:,:,mf]
            for nf in range(frameNumF):
                
                xykf             = peakXYF[nf]
                kf               = xykf[0,2].astype(np.int32)  # frame number
                offsXYF          = offsetArray[:,kf,mf]
                
                # the offsets in the template/augmented data
                xf, yf           = xykf[:,0].astype(np.int32), xykf[:,1].astype(np.int32)
                
                # for matrix expansion - do it once
                xykf             = xykf.T
            
                for mt in range(n_labels):
                    if mf == mt:
                        continue
                    
                    peakXYT              = peakArray[mt]
                    frameNumT            = len(peakXYT)
                    for nt in range(frameNumT):   
                        
                        xykt             = peakXYT[nt]
                        kt               = xykt[0,2].astype(np.int32)  # frame number
                        offsXYT          = offsetArray[:,kt,mt]
                        
                        # difference in location according to the training set
                        diffXYTF         = offsXYT - offsXYF
                        
                        # difference between all the detected points - matrices
                        diffX              = xykt[:,0].reshape((-1,1)) - xykf[0,:].reshape((1,-1))
                        diffY              = xykt[:,1].reshape((-1,1)) - xykf[1,:].reshape((1,-1))

                        # in the neighborhood
                        minDistance      = self.MIN_MATCH_DISTANCE # pixels
                        mtxDist          = (diffX - diffXYTF[0])**2 + (diffY - diffXYTF[1])**2
                        iy               = np.nonzero(mtxDist.min(axis=0) < minDistance)  
                        corrAccumF[yf[iy],xf[iy]] += 1
                        
                hasMatch         = np.any(corrAccumF[yf,xf] >= self.HITS_THRESHOLD)
                if hasMatch:
                    self.Print('Match label %d had %d label match' %(lblIds[mf],self.HITS_THRESHOLD))   
            
            # points are counted
            corrAccum[:,:,mf] = corrAccumF  
                
                # check boundaries
                #yt              = np.minimum(np.maximum(yt, 0), hp-1).astype(np.int32)
                #xt              = np.minimum(np.maximum(xt, 0), wp-1).astype(np.int32)
#                    if np.any(xt < 0) or np.any(yt < 0) or np.any(xt >= wp) or np.any(yt >= hp):
#                        continue
#                    # assign points with weight
#                    corrAccumT[yt,xt] += corrRespF[yf,xf]



                
#                tmpData = np.stack((corrRespF,corrRespT,corrAccumT),axis = 2)
#                self.ShowFeatures(tmpData, m)
#                
#           
        elapsed_time = time.time() - start_time
        self.Print('Counting in %f sec' %(elapsed_time))        
        return corrAccum      
    
    def ComputePredictedLabelPositionsOld(self, corrData, offsetArray, lblIds):
        # predict point position using mutual info
        #img               = self.ImagePreprocessing(img)
        start_time          = time.time()
        hp, wp, n_frames, n_labels  = corrData.shape
        dp, m_frames, m_labels      = offsetArray.shape
            
#        # the frames were augmented by angles and scale
#        warp_angles         = self.WARP_ANGLES
#        warp_scales         = self.WARP_SCALES
#        warp_num            = len(warp_angles)*len(warp_scales)
#        
#        # will result in bug - need to understand how angles and scales has beeen created
#        assert len(warp_scales) < 2         
#        file_num_f          = n_labels/warp_num
#        file_num            = int(file_num_f)            
#        if (file_num_f - file_num) != 0:
#            self.Print('Augmentation problem','E')
#            return
#        
#        warp_angles_ind     = np.remainder(np.arange(n_frames), warp_num)
#        warp_angles_rad     = [np.deg2rad(warp_angles[k]) for k in warp_angles_ind]
        
        # hit accumulator
        corrAccum           = np.zeros((hp, wp, n_frames))
        #inverseReference    = np.zeros((hp, wp, n_frames, n_labels))
        
        for k in range(n_frames):
            
            corrAccumT      = corrAccum[:,:,k]
            for m in range(n_labels):
                
                corrRespF            = corrData[:,:,k,m]
                offsXYF              = offsetArray[:,k,m]
                yf,xf                = np.nonzero(corrRespF > 0.5)
                if len(yf) < 1:
                    continue
                
                for n in range(n_labels):
                    
                    # project to 
                    corrRespT        = corrData[:,:,k,n]
                    ys,xs            = np.nonzero(corrRespT > 0.5) # debug
                    if len(ys) < 1:
                        continue
                    
                    offsXYT          = offsetArray[:,k,n]
                    # the offsets in the template/augmented data
                    vectXY           = offsXYT - offsXYF
                    yt, xt           = yf + vectXY[1], xf + vectXY[0]
                    yt, xt           = yt.astype(np.int32), xt.astype(np.int32)
                    
                    # check boundaries
                    #yt              = np.minimum(np.maximum(yt, 0), hp-1).astype(np.int32)
                    #xt              = np.minimum(np.maximum(xt, 0), wp-1).astype(np.int32)
#                    if np.any(xt < 0) or np.any(yt < 0) or np.any(xt >= wp) or np.any(yt >= hp):
#                        continue
#                    # assign points with weight
#                    corrAccumT[yt,xt] += corrRespF[yf,xf]
                    
                    # in the neighborhood
                    minDistance      = self.MIN_MATCH_DISTANCE # pixels
                    mtxDist          = (xt.reshape((-1,1)) - xs.reshape((1,-1)))**2 + (yt.reshape((-1,1)) - ys.reshape((1,-1)))**2
                    #iy,ix            = np.nonzero(mtxDist < minDistance)                    
                    #iy               = np.nonzero(mtxDist.min(axis=1) < minDistance)  
                    ix               = mtxDist.argmin(axis=1)
                    iy               = np.nonzero(mtxDist[range(len(ix)),ix] < minDistance)  
                    if len(iy) > 0:
                        # only the rows with hits
                        corrAccumT[yf[iy],xf[iy]] += 1
                        

                hasMatch         = np.any(corrAccumT[yf,xf] >= self.HITS_THRESHOLD)
                if hasMatch:
                    self.Print('Image %d - match of label %d' %(k,lblIds[m]))
                 
            # points are counted
            corrAccum[:,:,k] = corrAccumT  

                
#                tmpData = np.stack((corrRespF,corrRespT,corrAccumT),axis = 2)
#                self.ShowFeatures(tmpData, m)
#                
#           
        elapsed_time = time.time() - start_time
        self.Print('Counting in %f sec' %(elapsed_time))        
        return corrAccum      
      
    def ComputeMatchAndPredictions(self, imgIn, lblIds):
        # matching several labels
        # precompute the filters
        n_labels   = len(lblIds)
        n_r,n_c    = self.TEMPLATE_SIZE,self.TEMPLATE_SIZE
        n_frames   = len(self.templateList)
        
        #self.HITS_THRESHOLD           = n_labels - 1
        
        coeffArray = np.zeros((n_r,n_c,n_frames,n_labels))
        offsetArray= np.zeros((2,n_frames,n_labels))
        for k in range(n_labels):
            lblId                     = lblIds[k]
            tempData,tempOffs         = self.GetTemplatesPerLabel(lblId)
            coeffArrayTmp             = self.ComputeFiltersAugmented(tempData)                
            coeffArray[:,:,:,k]       = coeffArrayTmp
            offsetArray[:,:,k]        = tempOffs
                
        
        corrData                  = self.ComputeCorrelationAugmentedCoeff(imgIn, coeffArray)
        
        # count hits :  accumData - nr+template_size-1, nc+template_size-1, n_frames
        accumData                 = self.ComputePredictedLabelPositions(corrData, offsetArray, lblIds)
        
        # find hits to show them in the image
        iy,ix,ilbl                  = np.nonzero(accumData >= self.HITS_THRESHOLD)
        
        # show
        lblNames                      = ilbl
        lblXY                         = np.vstack((ix + 3, iy + 3 ))
        self.ShowImageLabelData(imgIn, lblXY, lblNames, 'Hits' )
        
#        for lblId in lblIds:
#            
#            #coeffArray                = self.ComputeFilters(tempData)
#            #score, peak_k, peak_c     = self.ComputeCorrelationWarpedTemplates(imgIn, coeffArray)
#            
#            corrData                  = self.ComputeCorrelationAugmentedCoeff(imgIn, coeffArray)
            
            
            #tempOffsets               = self.offsetList[peak_k] if score > 0.8 else self.offsetList[0]
        
            #self.ShowTemplatesPerLabel(49,lblId)
            #self.ShowFilters(coeffArray, level = lblId)
            #self.ShowImageLabelData(imgIn, tempOffsets, self.labelNames, 'Predicted - %d' %lblId)
        #self.ShowFeatures(accumData, 0)
            
        return True

        
#%% ----------------------------------------------------------------------------------    

    def ShowData(self, img = None, name = '', fnum = 31):
        # not debug mode - do nothing
        if not self.debugOn:
            return
        
        if img is None:
            self.Print('Load image data first', 'E')
            return
        
#        plt.figure(fnum)
#        plt.subplot(1,1,1),plt.imshow(img, cmap = 'brg')
#        plt.title(name), #plt.xticks([]), plt.yticks([])
#        plt.show()          
            
        cv.imshow("Test image %s" %name, img)
        cv.waitKey(100)
        return 

    def ShowImageLabelData(self, imgData = None, lblData = None, lblNames = None, name = ''):
        # not debug mode - do nothing
        if not self.debugOn:
            return
       
        if imgData is None:
            self.Print('Load image data first', 'E')
            return
        frame   = imgData.copy()    
        
        if lblData is None:
            self.Print('Load label data first', 'E')
            return
        
        if len(lblData) < 1:
            self.Print('Label data is not found and not will be shown', 'W')
            
        # make row vector
        if lblData.shape[0] > lblData.shape[1]:
            lblData = lblData.T
            
        if lblNames is None:
            lblNames = list(range(lblData.shape[1]))
            
        
        objId = 0 #self.objectId 
        if objId < 0:
            self.Print('Object data is not initialized well', 'W')
                    
        frameH, frameW  = frame.shape[:2]
        fs, ft, fc, ff = 0.6, 2, (240,220,10),  cv.FONT_HERSHEY_SIMPLEX
        #color   = list(np.random.random(size=(obj_num,3)) * 256)
        clr     = (250, 0, 250)
        for k in range(len(lblNames)):
            pLabels = lblNames[k]
            pXY     = lblData[:,k]
             #color[int(objId)-1]
            
            x,y         = int(pXY[0]),int(pXY[1])
            if x < 0 or x > frameW or y < 0 or y > frameH:
                self.Print('Object %s point %d is out of range (%d, %d)' %(objId,k,x,y),'I')
                continue
            
            frame       = cv.circle(frame, (x,y) ,3, (0, 0, 250),-1,8)
            frame       = cv.putText(frame ,'{:s}-{:s}'.format(str(objId),str(pLabels)) ,(x+20,y), ff, fs, clr, ft,cv.LINE_AA) 
        
        #frame       = cv.putText(frame ,'{:d}-{:s}'.format(objId,str(pLabels[ii])) ,(x+20,y), ff, fs,color,ft,cv.LINE_AA)
        # for schunk
        while frame.shape[0] > 2400:
            frame       = cv.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        #...and finally display it

        cv.imshow("%s : I and L " %name, frame)
        cv.waitKey()
        #cv.destroyAllWindows()
        return
    
    def ShowImageAndDataXY(self, imgData = None, dataXY = None, name = ''):
        # not debug mode - do nothing
        if not self.debugOn:
            return
       
        if imgData is None:
            self.Print('Load image data first', 'E')
            return
        frame   = imgData.copy()    
        
        if dataXY is None:
            self.Print('Load label data first', 'E')
            return
        
        if len(dataXY) < 1:
            self.Print('Label data is not found and not will be shown', 'W')
            
        # make row vector
        if dataXY.shape[1] == 2:
            dataXY = dataXY.T
        
        objId = 0 #self.objectId 
        if objId < 0:
            self.Print('Object data is not initialized well', 'W')
                    
        frameH, frameW  = frame.shape[:2]
        fs, ft, fc, ff = 0.6, 2, (240,220,10),  cv.FONT_HERSHEY_SIMPLEX
        #color   = list(np.random.random(size=(obj_num,3)) * 256)
        clr     = (250, 0, 250)
        for k in range(dataXY.shape[1]):
            pXY     = dataXY[:,k]
             #color[int(objId)-1]
            
            x,y         = int(pXY[0]),int(pXY[1])
            if x < 0 or x > frameW or y < 0 or y > frameH:
                self.Print('Object %s point %d is out of range (%d, %d)' %(objId,k,x,y),'I')
                continue
            
            frame       = cv.circle(frame, (x,y) ,3, (0, 0, 250),-1,8)
            #frame       = cv.putText(frame ,'{:s}-{:s}'.format(str(objId),str(pLabels)) ,(x+20,y), ff, fs, clr, ft,cv.LINE_AA) 
        
        #frame       = cv.putText(frame ,'{:d}-{:s}'.format(objId,str(pLabels[ii])) ,(x+20,y), ff, fs,color,ft,cv.LINE_AA)
        # for schunk
        while frame.shape[0] > 2400:
            frame       = cv.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
        #...and finally display it

        cv.imshow("%s : I and L " %name, frame)
        cv.waitKey(100)
        #cv.destroyAllWindows()
        return

    
    def ShowTemplate(self, tempData, name = ''):
        # debug - show template data
        if not self.debugOn:
            return
        
        # failed to match
        if len(tempData) < 1:
            return
        
        # tempData - NxNxDxL
        templateSide,x,dim, labelNum = tempData.shape
        
        montage_size    = int(np.ceil(np.sqrt(labelNum)))
        images          = np.zeros((montage_size*templateSide,montage_size*templateSide,dim), dtype = tempData.dtype)
        for r in range(montage_size):
            for c in range(montage_size):
                ri , ci = r*templateSide, c*templateSide
                k  = r*montage_size + c
                if k < labelNum:
                    images[np.int32(ri):np.int32(ri+templateSide), np.int32(ci):np.int32(ci+templateSide),:] = tempData[:,:,:,k]
        # too small
        while images.shape[1] < 500:
            images       = cv.resize(images,(int(images.shape[1]/0.5),int(images.shape[0]/0.5)))
        
        # construct the montages for the images
        cv.imshow("M %s" %name, images)
        cv.waitKey(100)
        return    
    
    def ShowTemplatesPerLabel(self, showNum = 10, labelId = 1):
        # show object data
        
        frameNum      = len(self.templateList)
        if frameNum < 1:
            self.Print('Load object training data','E')
            return
        
        templateSide, x, dim, labelNum      = self.templateList[0].shape
        if labelNum < 5:
            self.Print('Label data is not correct - must have at least 6-7 labels','E')
            return 
        
        # limit the number of data to show
        showNum         = np.minimum(showNum,frameNum)
        
        # container
        tempData     = np.zeros((templateSide,templateSide,dim,showNum),dtype=np.uint8)
        #num_list     = random.sample(range(0, frameNum), showNum)
        num_list     = [k for k in range(showNum)]
        
        for k in range(showNum): 
            n           = num_list[k]
            tmpObjData  = self.templateList[n]
            #self.objectList[n].ShowData()
            tempData[:,:,:,k] = tmpObjData[:,:,:,labelId]
            
        self.ShowTemplate(tempData,' Label: %d' %(labelId))
        #self.ShowTemplate(tempData,' Label')
        #self.Print(str(num_list))           
        return 
    
    def ShowFilters(self, filt, level = 0):
        """Helper function to plot a gallery of portraits"""
        if not self.debugOn:
            return
        
        h,w, chanNum, filtNum = filt.shape
        n_row = filtNum
        n_col = chanNum
        #plt.figure(70+level, figsize=(1.8 * n_col, 2.4 * n_row))
        plt.figure('%d:L%d-Filters' %(self.figNum,level), figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
        for f in range(filtNum):
            for c in range(chanNum):
                plt.subplot(n_row, n_col, f*chanNum + c + 1)
                plt.imshow(filt[:,:,c,f].reshape((h, w)), cmap=plt.cm.gray)
                plt.title('L%d-F%d,C%d' %(level,f,c), size=10)
                plt.xticks(())
                plt.yticks(())
    
        plt.show()
        
    def ShowFeatures(self, filt, level = 0):
        """Helper function to plot a gallery of features"""
        #fb2_montage = montage(fb2, rescale_intensity=True)
        if not self.debugOn:
            return
        
        if len(filt.shape) < 4:
            filt = np.expand_dims(filt,axis=3)
        
        h,w, chanNum, filtNum = filt.shape
        n_row = filtNum  
        n_col = chanNum
        #plt.figure(70+level, figsize=(1.8 * n_col, 2.4 * n_row))
        plt.figure('%d:L%d-Features' %(self.figNum,level), figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
        for f in range(filtNum):
            for c in range(chanNum):
                plt.subplot(n_row, n_col, f*chanNum + c + 1)
                plt.imshow(filt[:,:,c,f].reshape((h, w)), cmap=plt.cm.gray)
                #plt.imshow(filt[:,:,c].reshape((h, w)), cmap=plt.cm.gray)
                plt.title('L%d-F%d,C%d' %(level,c,f), size=10)
                plt.xticks(())
                plt.yticks(())
    
        plt.show()    
        
    def ShowWarpedTemplates(self, tempData, name = ''):
        # debug - show template data
        if not self.debugOn:
            return
        
        # failed to match
        if len(tempData) < 1:
            return
        
        # tempData - NxNxDxL
        templateSide,x,labelNum = tempData.shape
        
        montage_size    = int(np.ceil(np.sqrt(labelNum)))
        images          = np.zeros((montage_size*templateSide,montage_size*templateSide), dtype = tempData.dtype)
        for r in range(montage_size):
            for c in range(montage_size):
                ri , ci = r*templateSide, c*templateSide
                k  = r*montage_size + c
                if k < labelNum:
                    images[np.int32(ri):np.int32(ri+templateSide), np.int32(ci):np.int32(ci+templateSide)] = tempData[:,:,k]
        # too small
        while images.shape[1] < 500:
            images       = cv.resize(images,(int(images.shape[1]/0.5),int(images.shape[0]/0.5)))
            
        while images.shape[1] > 2000:
            images       = cv.resize(images,(int(images.shape[1]*0.75),int(images.shape[0]*0.75)))
            
        # construct the montages for the images
        cv.imshow("M %s" %name, images)
        cv.waitKey(100)
        return          
        
    def Finish(self):
        
        cv.destroyAllWindows() 
        self.Print('L-Manager is closed')
        
    def Print(self, txt='',level='I'):
        
        if level == 'I':
            ptxt = 'I: LM: %s' % txt
            logging.info(ptxt)  
        if level == 'W':
            ptxt = 'W: LM: %s' % txt
            logging.warning(ptxt)  
        if level == 'E':
            ptxt = 'E: LM: %s' % txt
            logging.error(ptxt)  
           
        print(ptxt)
        
       
#%% --------------------------           
class TestLabelMatchManager(unittest.TestCase):                
    def test_Create(self):
        d       = LabelMatchManager()
        self.assertEqual(False, d.state)
        
    def test_LoadTrainingFiles(self):
        # check read training files  
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0006'
        d           = LabelMatchManager()
        isOk        = d.LoadTrainingFiles(filePath)
        self.assertTrue(isOk)
        
    def test_ConvertFilesToObjects(self):
        # check read training files  
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0007'

        d           = LabelMatchManager()
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjects()
        k           = 1
        d.ShowImageLabelData(d.frameList[k], d.offsetList[k], d.labelNames, '%d' %k)
        self.assertTrue(isOk)
        
    def test_ConvertFilesToObjectsWithAugmentation(self):
        # check read training files and augment the data  
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0007'

        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 2
        
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjectsWithAugmentation()
        k           = 1
        d.ShowImageLabelData(d.frameList[k], d.offsetList[k], d.labelNames, '%d' %k)
        self.assertTrue(isOk)        
        

    def test_ShowTemplatesPerLabel(self):
        # shows data per label 
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0007'
   
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 49
        lblId       = 12

        isOk        = d.LoadTrainingFiles(filePath)
        #isOk        = d.ConvertFilesToObjects()
        isOk        = d.ConvertFilesToObjectsWithAugmentation()
        tempData    = d.GetTemplatesPerLabel(lblId)
        d.ShowTemplatesPerLabel(49,lblId)
        cv.waitKey()
        self.assertTrue(isOk)        


    def test_ComputeDecomposition(self):
        # not applicable
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0007'
        lblId       = 7      
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 64

        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjects()
        tempData,tempOff    = d.GetTemplatesPerLabel(lblId)
        flt         = d.ComputeDecomposition(tempData)
        
        d.ShowTemplatesPerLabel(49,lblId)
        d.ShowFilters(flt)
        cv.waitKey()
        
    def test_ComputeCorrelation(self):
        # shows data per label 
        filePath    = r'C:\Users\udubin\Documents\Code\RAI\Objects\CupRGB-02\videos\object_0007'

        lblId       = 1
        imgId       = 2
        
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 8
        d.TEMPLATE_SIZE       = 32  # size of the template for labels
        
        
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjects()
        tempData, tempOffset    = d.GetTemplatesPerLabel(lblId)
        #coeffArray   = d.ComputeDecomposition(tempData)
        coeffArray   = tempData #
        
        imgArray    = d.frameList[imgId]
        #corrData    = d.ComputeCorrelation(imgArray, coeffArray)
        corrData    = d.ComputeCorrelationTemplates(imgArray, coeffArray)
        
        print(d.offsetList[imgId][:,lblId])
        
        d.ShowData(imgArray)
        d.ShowTemplatesPerLabel(49,lblId)
        d.ShowFilters(coeffArray)
        d.ShowFeatures(corrData)
        
    def test_ComputeWarpedCorrelation(self):
        # shows data per label 
        #filePath    = r'D:\RobotAI\Customers\TED\Objects\object1\labels'
        #filePath    = r'D:\RobotAI\Customers\TM\Objects\MaskBox-03\labels'
        #filePath    = r'D:\RobotAI\Customers\TM\Objects\MaskBox-03\labels\mask_box_up'
        #filePath    = r'D:\RobotAI\Customers\Moona\Objects\PlugUSB-IDS\videos\object_01'
        filePath    = r"D:\RobotAI\Customers\Shiba\Objects\Scissors_04\videos\object_0000"

        lblId       = 1
        imgId       = 13
        
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 8
        d.TEMPLATE_SIZE       = 16  # size of the template for labels
        
        
        isOk        = d.LoadTrainingFiles(filePath)
        #isOk        = d.ConvertFilesToObjects()
        isOk        = d.ConvertFilesToObjectsWithAugmentation()

        tempData    = d.GetTemplatesPerLabel(lblId)
        
        #coeffArray  = d.ComputeFilters(tempData)
        coeffArray  = d.ComputeFiltersAugmented(tempData)
        
        imgArray    = d.frameList[imgId]
        #imgArray    = cv_warp(imgArray,0,0,5,1)
        #corrData    = d.ComputeCorrelation(imgArray, coeffArray)
        corrData    = d.ComputeCorrelationWarpedTemplates(imgArray, coeffArray)
        
        print(d.offsetList[imgId][:,lblId])
        
        d.ShowData(imgArray)
        d.ShowTemplatesPerLabel(49,lblId)
        d.ShowFilters(coeffArray)
        #d.ShowFeatures(corrData)        
        
        

    def test_ComputeImageDistance(self):
        # shows data per label 
        #filePath    = r'D:\RobotAI\Customers\TED\Objects\object1\labels'
        #filePath    = r'D:\RobotAI\Customers\TM\Objects\MaskBox-03\labels'
        #filePath    = r'D:\RobotAI\Customers\TM\Objects\MaskBox-03\labels\mask_box_up'
        filePath    = r'D:\RobotAI\Customers\Moona\Objects\PlugUSB-IDS\videos\object_01'
        lblId       = 3
        
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 9
        d.TEMPLATE_SIZE       = 32  # size of the template for labels
        
        
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjects()
        tempData    = d.GetTemplatesPerLabel(lblId)
        img         = d.ComputeImageDistance(tempData)
        
        d.ShowTemplatesPerLabel(49,lblId)
        
    def test_ComputeMultiLabelMatch(self):
        # matching several labels
        filePath    = r"D:\RobotAI\Customers\Shiba\Objects\Scissors_04\videos\object_0000"

        lblIds      = [0,2,9]
        imgId       = 2
        
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 8
        d.TEMPLATE_SIZE       = 16  # size of the template for labels
               
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjects()
        
        imgArray    = d.frameList[imgId]
        imgArray    = cv_warp(imgArray,0,0,-5,1)
        
        
        for lblId in lblIds:
            tempData    = d.GetTemplatesPerLabel(lblId)
            coeffArray  = d.ComputeFilters(tempData)
            corrData    = d.ComputeCorrelationWarpedTemplates(imgArray, coeffArray, lblId)
            
            print('Actual Offsets:')
            print(d.offsetList[imgId][:,lblId])
        
            d.ShowTemplatesPerLabel(49,lblId)
            d.ShowFilters(coeffArray, level = lblId)
            
        #d.ShowFeatures(corrData)
        d.ShowData(imgArray)
        
    def test_ComputeMatchAndPredictions(self):
        # matching several labels and predicting the rest
        filePath    = r"D:\RobotAI\Customers\Shiba\Objects\Scissors_04\videos\object_0000"
        imgPath     = r"D:\RobotAI\Customers\Shiba\Objects\Scissors_04\videos\object_0001\00000011.jpg"

        lblIds      = [0,2,9]
        imgId       = 3
        
        d           = LabelMatchManager()
        d.MAX_FILES_TO_LOAD   = 8
        d.TEMPLATE_SIZE       = 16  # size of the template for labels
        d.WARP_ANGLES         = [0, 10,-10, 20,-20,-30, 30]
        d.MIN_MATCH_DISTANCE  = 20**2
        d.CORR_THRESHOLD      = 0.9
        d.HITS_THRESHOLD      = 2
               
        isOk        = d.LoadTrainingFiles(filePath)
        isOk        = d.ConvertFilesToObjectsWithAugmentation()
        
        imgArray    = d.frameList[imgId]
        imgArray    = cv_warp(imgArray,0,0,5,1)        
        imgArray    = cv.imread(imgPath)
        
        isOk        = d.ComputeMatchAndPredictions(imgArray, lblIds)
        
        #d.ShowData(imgArray)
        self.assertTrue(isOk)        

        
         
        
        

#%%
if __name__ == '__main__':
    #print (__doc__)
    
    #unittest.main()
    
    # template manager test
    singletest = unittest.TestSuite()
    #singletest.addTest(TestLabelMatchManager("test_Create"))
    #singletest.addTest(TestLabelMatchManager("test_LoadTrainingFiles")) # ok
    #singletest.addTest(TestLabelMatchManager("test_ConvertFilesToObjects")) # ok
    #singletest.addTest(TestLabelMatchManager("test_ConvertFilesToObjectsWithAugmentation")) # ok       
    #singletest.addTest(TestLabelMatchManager("test_ShowTemplatesPerLabel")) # ok
    
    singletest.addTest(TestLabelMatchManager("test_ComputeDecomposition"))
    #singletest.addTest(TestLabelMatchManager("test_ComputeCorrelation")) # 
    #singletest.addTest(TestLabelMatchManager("test_ComputeImageDistance"))
    #singletest.addTest(TestLabelMatchManager("test_ComputeWarpedCorrelation")) # ok
    #singletest.addTest(TestLabelMatchManager("test_ComputeMultiLabelMatch")) # ok
    #singletest.addTest(TestLabelMatchManager("test_ComputeMatchAndPredictions")) # 


    unittest.TextTestRunner().run(singletest)
 