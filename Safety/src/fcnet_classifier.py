
'''

ROI Classifier - classifies multiple patches and ROIs using NNet - FCNET.
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
from fcnet.train import Model

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from common import log

#%% Help
def to_categorical(x, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]

    >>> b = np.array([.9, .04, .03, .03,
    ...               .3, .45, .15, .13,
    ...               .04, .01, .94, .05,
    ...               .12, .21, .5, .17],
    ...               shape=[4, 4])
    >>> loss = keras.ops.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = keras.ops.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

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

        #INPUT_SIZE      = 784
        HIDDEN_SIZE     = [32, 32]
        OUTPUT_SIZE     = 2   
        dim_input       = self.X_train.shape[1]
        if self.config is None:
            self.config     = [HIDDEN_SIZE[0], HIDDEN_SIZE[1], OUTPUT_SIZE]
       
        # Create the Neural Network model
        self.model = Model(input_size=dim_input, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
        
    
        #print(self.model)
        self.tprint("Model input dim : %s" %str(dim_input))
        self.tprint("Model config : %s" %str(self.config))

    def set_data(self, xt, yt, xv, yv):
        "assign data to internal vars"
        self.X_train, self.y_train, self.X_test, self.y_test  = xt, yt, xv, yv
        ret = self.check_data()
        return ret

    def check_data(self):
        "make sure that data is ok"
        ret1 = np.all(np.isin(self.y_train, [0,1]))
        ret2 = np.all(np.isin(self.y_test,  [0,1]))
        ret3 = self.X_train.shape[0] == self.y_train.shape[0]
        self.tprint('Min max range of the data : %s-%s' %(str(self.X_train.min()), str(self.X_train.max())))
        return ret1 and ret2 and ret3

    def train(self):
        "train the model"
        # must have data loadded already
        ret = self.check_data()
        if not ret:
            self.tprint('Label data must be 0 and 1')
            return

        # create a new model
        if self.model is None:
            self.create_model()

        # make data categorical - 2 columns
        y_train_categorical = to_categorical(self.y_train,2)

        # optimization
        self.model.train(self.X_train, y_train_categorical, initial_learning_rate=0.001, decay=0.001, n_epochs=500, plot_training_results=True)

        self.tprint('Training is done!')
        
        # prediction
        y_pred = self.predict(self.X_train)
        self.tprint('Training accuracy:')
        acc    = self.check_accuracy(self.y_train, y_pred)

        return True

    def predict(self, X, y = None):
        "make model prediction"
        time_s  = time.time()

        # calculate the forward pass  
        output  = self.model.forward(inputs=X)
        
        # Calculate the loss (Categorical Crossentropy)
        #epsilon = 1e-10
        #loss    = -np.mean(y * np.log(output + epsilon))
        
        # calculate the accuracy 
        y_pred  = np.argmax(output, axis=1)        

        #y_pred  = np.array([s.data > 0 for s in scores])

        self.tprint('Mean predict time : %s sec' %str((time.time() - time_s)/X.shape[0]))

        # if y is given - check accuracy
        if y is not None:
            acc = self.check_accuracy(y, y_pred)

        return y_pred
    
    def check_accuracy(self, y, y_pred):
        #
        # IOU = true_positive / (true_positive + false_positive + false_negative).
        #num_classes         = 2
        mres    = np.mean(y_pred == y.flatten())
        self.tprint("Mean accuracy = %s" %str(mres*100))
        return mres


    def save_model(self):
        "saves the model/weight to a file"
        sfile = 'model.dat'
        model_params = {
                        'model':  self.model,
                        'config': self.config
                        }

        fileObj = open(sfile, 'wb')
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

        elif data_type == 2: # 2D dataset
            # check that all are compatible
            point_num           = 30
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
            y                  = y #*2 - 1

        else:
            # check that all are compatible 8 D
            point_num           = 200
            dim_num             = 64
            X00                 = np.random.rand(point_num,dim_num)
            y00                 = np.zeros((point_num,1))
            X11                 = np.random.rand(point_num,dim_num)+np.array([1]*dim_num)
            y11                 = np.zeros((point_num,1))
            X10                 = np.random.rand(point_num,dim_num)+np.array([[1,0]*int(dim_num/2)])
            y10                 = np.ones((point_num,1))
            X01                 = np.random.rand(point_num,dim_num)+np.array([[0,1]*int(dim_num/2)])
            y01                 = np.ones((point_num,1))  

            X                   = np.vstack((X00,X01,X10,X11))
            y                   = np.vstack((y00,y01,y10,y11))

            # normalize the data to -1:1 and msks to 0:1
            X                  = X - 1
            y                  = y #*2 - 1
            
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

    def test_simple(self, test_type = 2):
        # test model on 2D XOR data
        
        self.X_train, self.y_train, self.X_test, self.y_test  = self.test_create_dataset(test_type)

        self.create_model()
        self.train()

        time_s                      = time.time()
        y_pred                      = self.predict(self.X_test)
        self.tprint('Mean predict time : %s sec' %str((time.time() - time_s)/self.X_test.shape[0]))

        # check IoU
        self.check_accuracy(self.y_test, y_pred)

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
        ret     = p.test_simple(2)
        #p.save_model()
        p.finish()
        self.assertTrue(ret)


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

    def test_train_small_dataset(self):
        "data set from robot in motion - train and test"
        d               = DataSource()
        d.patch_size    = (16, 16) 
        d.patch_step    = 2
        file_num        = 1

        dirpath1         = r'C:\Users\udubin\Documents\Projects\Safety\data\laser_classifier\small\off'
        dirpath2         = r'C:\Users\udubin\Documents\Projects\Safety\data\laser_classifier\small\on'
        dir_paths        = [dirpath1, dirpath2]
        mask_values      = [0,1]
        rois             = [(400,500,460,560)] #, (800,300,860,360)]
        xt,yt,xv,yv      = d.create_dataset(dir_paths, file_num, rois, mask_values)

        p                = RoiClassifier()
        ret              = p.set_data(xt,yt,xv,yv)
        ret              = p.train()
        ret              = p.predict(xv,yv)
        
        self.assertTrue(xv.shape[0] == yv.shape[0])

    def test_train_static_dataset(self):
        "data set from robot in motion - train and test"
        d               = DataSource()
        d.patch_size    = (16, 16) 
        file_num        = 1

        dirpath1         = r'C:\Data\Safety\AGV\12_static_both_prj_covered_hall_carpet\12_static_both_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        dirpath2         = r'C:\Data\Safety\AGV\12_static_no_prj_covered_hall_carpet\12_static_no_prj_covered_hall_carpet\device_0_sensor_0_Infrared_1_image_data'
        dir_paths        = [dirpath1, dirpath2]
        mask_values      = [0,1]
        rois             = [(400,500,460,560)]
        xt,yt,xv,yv      = d.create_dataset(dir_paths, file_num, rois, mask_values)

        p                = RoiClassifier()
        ret              = p.set_data(xt,yt,xv,yv)
        ret              = p.train()
        ret              = p.predict(xv,yv)
        
        self.assertTrue(xv.shape[0] == yv.shape[0])        
               



# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestRoiClassifier("test_save_load")) # nok
    #suite.addTest(TestRoiClassifier("test_data_show")) # ok
    #suite.addTest(TestRoiClassifier("test_train_simple")) # ok 
    suite.addTest(TestRoiClassifier("test_train_small_dataset")) #   
    #suite.addTest(TestRoiClassifier("test_train_static_dataset")) 

    runner = unittest.TextTestRunner()
    runner.run(suite)

#%%
if __name__ == '__main__':
    #print (__doc__)
    #r = RoiClassifier()
    #r.test_simple()
    RunTest()
    

   


    
 

