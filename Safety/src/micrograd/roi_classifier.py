
'''

ROI Classifier - classifies multiple patches and ROIs using NNet.
Each patch is marked 0 or 1 according to the folder with laser Off or On.

Support training and testing

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
import unittest
import time
import pickle
import os

#from skimage.util.shape import view_as_windows
#from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

from dataset.image_dataset import DataSource
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from common import log



# --------------------------------
#%% Main Object
class RoiClassifier:

    def __init__(self):

        # params
        self.model           = None  # nnet model 
        self.config          = [16, 16, 1]  # nnet config layers
        
        self.X_train         = None 
        self.y_train         = None
        self.X_test          = None 
        self.y_test          = None        

        self.tprint('Classifier is defined')

    def create_model(self):
        "creates simple NN model"
        # initialize a model 
        self.config     = [16, 16, 1]
        self.model      = MLP(2, self.config) # 2-layer neural network
        #print(self.model)
        self.tprint("Model config : %s" %str(self.config))
        self.tprint("Model number of parameters", len(self.model.parameters()))


    def check_data(self):
        "make sure that data is ok"
        ret1 = np.any(np.abs(self.y_train) > 0.9) # not zeros
        ret2 = np.any(np.abs(self.y_train) < 1.1) # only 1 and -1
        return ret1 and ret2

    def loss(self, batch_size=None):
        # loss function
    
        # inline DataLoader :)
        if batch_size is None:
            Xb, yb = self.X_train, self.y_train
        else:
            ri = np.random.permutation(self.X_train.shape[0])[:batch_size]
            Xb, yb = self.X_train[ri], self.y_train[ri]

        inputs = [list(map(Value, xrow)) for xrow in Xb]
        
        # forward the model to get scores
        scores = list(map(self.model, inputs))
        
        # svm "max-margin" loss
        losses      = [(1 + Value(-yi)*scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss   = sum(losses) * (1.0 / len(losses))
        # L2 regularization
        alpha = 1e-4
        reg_loss = alpha * sum((p*p for p in self.model.parameters()))
        total_loss = data_loss + reg_loss
        
        # also get accuracy
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    def train(self):
        "train the model"
        # optimization
        ret = self.check_data()
        if not ret:
            self.tprint('Label data must be -1 and 1')
            return
        
        for k in range(100):
            
            # forward
            total_loss, acc = self.loss()
            
            # backward
            self.model.zero_grad()
            total_loss.backward()
            
            # update (sgd)
            learning_rate = 1.0 - 0.9*k/100
            for p in self.model.parameters():
                p.data -= learning_rate * p.grad
            
            if k % 5 == 0:
                self.tprint(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    def predict(self, X):
        "make model prediction"
        inputs  = [list(map(Value, xrow)) for xrow in X]
        scores  = list(map(self.model, inputs))
        Z       = np.array([s.data > 0 for s in scores])
        return Z

    def save_model(self):
        "saves the model/weight to a file"
        model_params = {
                        'model':  self.model,
                        'config': self.config
                        }
        

        fileObj = open('model.dat', 'wb')
        pickle.dump(model_params,fileObj)
        fileObj.close()
        return True

    def load_model(self):
        "load the model from existing file"
        sfile = 'model.dat'
        ret = os.path.exists(sfile)
        if not ret:
            self.tprint('File {sfile} not found')
            return ret
        fileObj = open(sfile, 'rb')
        model_params = pickle.load(fileObj)
        fileObj.close()
        print(model_params)   
        return True  

    def show_data(self, X, y):
        "plot the data"
        y = y*2 - 1 # make y be -1 or 1
        # visualize in 2D
        plt.figure(figsize=(5,5))
        plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
        plt.show()

    def show_decision_boundaries(self, X, y):
        "shows in 2D visualize decision boundary"

        h = 0.25
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]
        inputs = [list(map(Value, xrow)) for xrow in Xmesh]
        scores = list(map(self.model, inputs))
        Z = np.array([s.data > 0 for s in scores])
        Z = Z.reshape(xx.shape)

        fig = plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()    

    def check_accuracy(self, y, y_pred):
        #
        # IOU = true_positive / (true_positive + false_positive + false_negative).
        #num_classes         = 2
        accuracy = [(yi > 0) == (scorei > 0) for yi, scorei in zip(y, y_pred)]
        mres     = sum(accuracy) / len(accuracy)
        self.tprint("Mean accuracy = %s" %str(mres*100))
        return mres
        
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
    
    def test_create_dataset(self, data_type = 1):
        "create train and test datasets - simple one"

        if data_type == 1:
            X, y = make_moons(n_samples=100, noise=0.1)
            y = y*2 - 1 # make y be -1 or 1
        else:
            # check that all are compatible
            point_num           = 10
            X00                 = np.random.rand(point_num,2)
            y00                 = np.zeros((point_num,1))
            X11                 = np.random.rand(point_num,2)+1
            y11                 = np.zeros((point_num,1))
            X10                 = np.random.rand(point_num,2)+np.array([1,0])
            y10                 = np.ones((point_num,1))
            X01                 = np.random.rand(point_num,2)+np.array([0,1])
            y01                 = np.ones((point_num,1))  

            X                   = np.vstack((X00,X01,X10,X11))
            y                   = np.vstack((y00,y01,y10,y11))

            # normalize the data to -1:1 and msks to 0:1
            X                  = X - 1
            y                  = y*2 - 1
            
        self.tprint(f'Mak : minimal value : {y.min()}, maximal value {y.max()}')

        # split the data on 2 datasets
        data_num            = X.shape[0]
        split_num           = int(data_num*0.8)
        ri                  = np.random.permutation(data_num)
        x_train, y_train    = X[ri[:split_num],:], y[ri[:split_num]]
        x_test, y_test      = X[ri[split_num:],:], y[ri[split_num:]]

        self.tprint(f'Training point number : {x_train.shape[0]}')
        self.tprint(f'Testing point number  : {x_test.shape[0]}')
        return x_train, y_train, x_test, y_test 

    def test_simple(self):
        # test model on 2D XOR data
        
        self.X_train, self.y_train, self.X_test, self.y_test  = self.test_create_dataset(2)

        self.create_model()
        self.train()

        time_s                      = time.time()
        y_pred                      = self.predict(self.X_test)
        self.tprint('Mean predict time : %s sec' %str((time.time() - time_s)/self.X_test.shape[0]))

        # check IoU
        self.check_accuracy(self.y_test, y_pred)

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

    def finish(self):
        # Close down the video stream
        self.tprint('Done')

    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt)     

# --------------------------------        
#%% Tests
class TestRoiClassifier(unittest.TestCase):

    def test_save_load(self):
        "testing save and load"
        p       = RoiClassifier()
        ret     = p.load_model()
        self.assertFalse(ret)
        ret     = p.save_model()
        ret     = p.load_model()
        self.assertTrue(ret)

    def test_data_show(self):
        "show simple data"
        p       = RoiClassifier()
        p.X_train, p.y_train, p.X_test, p.y_test  = p.test_create_dataset(2)
        p.create_model()
        p.show_data(p.X_train,p.y_train)
        p.finish()
        self.assertFalse(p.X_train is None)        
  
    def test_train_simple(self):
        "train on simple data"
        p       = RoiClassifier()
        ret     = p.test_simple()
        p.save_model()
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
    #suite.addTest(TestRoiClassifier("test_save_load")) # ok
    #suite.addTest(TestRoiClassifier("test_data_show")) # ok
    suite.addTest(TestRoiClassifier("test_train_simple")) # ok    

    runner = unittest.TextTestRunner()
    runner.run(suite)

#%%
if __name__ == '__main__':
    #print (__doc__)
    #r = RoiClassifier()
    #r.test_simple()
    RunTest()
    

   


    
 

