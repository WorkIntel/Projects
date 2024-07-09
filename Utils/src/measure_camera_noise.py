
'''
Noise Measurement App
==================

Using depth image to measure nose in ROI.

Usage
-----

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\barcode

Install : 


'''


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# local modules
# importing common Use modules 
from common import RectSelector
import sys 
sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense



def draw_str(dst, target, s):
    x, y = target
    dst = cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    dst = cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    return dst

class NoiseApp:
    def __init__(self, src):
        self.cap = RealSense() #video.create_capture(src, presets['book'])
        self.cap.change_mode('dep')
        self.frame = None
        self.rect  = None
        self.paused = False
        self.start_over = False
        self.img_int_mean = None
        self.img_int_std  = None

        self.show_dict     = {} # hist show

        cv.namedWindow('Noise')
        self.rect_sel = RectSelector('Noise', self.on_rect)

    def define_roi(self, image, rect, data=None):
        '''define ROI target.'''
        x0, y0, x1, y1 = rect

    def statistics(self, frame):

        ii  = np.logical_and(frame > 0.5, frame < 15000) # 65535 - None

        isok = np.any(np.isfinite([frame[ii]]))
        minv = frame[ii].min()
        maxv = frame[ii].max()
        avg  = frame[ii].mean()
        stdv = frame[ii].std()
        locv = ii.sum()  #np.any(np.isnan(frame[ii]))
        print(f'Stat : {minv:.2f}:{avg:.2f}:{maxv:.2f} - std {stdv:.2f}3f. Has None {locv}, is finite {isok}')


    def on_rect(self, rect):
        "remember ROI defined by user"
        #self.define_roi(self.frame, rect)
        self.rect = rect
        self.start_over = True
        print('ud1')


    def process_image(self, img):
        "makes integration over ROI"
        if self.rect is None:
            print('define ROI')
            return 0
        
        x0, y0, x1, y1 = self.rect
        img_roi = img[y0:y1,x0:x1].astype(np.float32)

        if self.start_over or (self.img_int_mean is None):
            self.img_int_mean = img_roi
            self.img_int_std  = np.zeros_like(img_roi)
            self.start_over = False
        
        self.img_int_mean += 0.1*(img_roi - self.img_int_mean)
        self.img_int_std  += 0.1*(np.abs(img_roi - self.img_int_mean) - self.img_int_std)
        err_std       = self.img_int_std.mean()
        return err_std

    def show_scene(self, err_std):
        "draw scene and ROI"
        vis     = self.frame.copy()
        #vis = cv.cvtColor(self.frame, cv.COLOR_GRAY2RGB)
        vis     = cv.convertScaleAbs(self.frame, alpha=0.03)
        self.rect_sel.draw(vis)

        if self.rect is not None:
            x0, y0, x1, y1 = self.rect
            clr = (0, 0, 0) if vis[y0:y1,x0:x1].mean() > 128 else (240,240,240)
            vis = cv.rectangle(vis, (x0, y0), (x1, y1), clr, 2)
            vis = draw_str(vis,(x0,y0-10),str(err_std))

        return vis
    
    def show_histogram(self, img):
        "show roi histgram"
        if self.rect is None:
            print('define ROI')
            return 0
        
        x0, y0, x1, y1 = self.rect
        img_roi = img[y0:y1,x0:x1].astype(np.float32)
        # Compute histogram
        hist, bins = np.histogram(img_roi.flatten(), bins=1024, range=[0, 2**15])

        if not 'fig' in self.show_dict : #len(self.show_dict) < 1:
            fig, ax = plt.subplots()
            fig.set_size_inches([24, 16])
            ax.set_title('Histogram (Depth)')
            ax.set_xlabel('Bin')
            ax.set_ylabel('Frequency')
            lineGray, = ax.plot(bins[:-1], hist, c='k', lw=3)
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(0, max(hist)+10)
            plt.ion()
            #plt.show()

            self.show_dict = {'fig':fig, 'ax':ax, 'line':lineGray}
        else:
            self.show_dict['line'].set_ydata(hist)
        
        self.show_dict['fig'].canvas.draw()
        return

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()

            self.statistics(frame)
            err_std = self.process_image(frame)

            self.show_histogram(frame)
            vis     = self.show_scene(err_std)
            cv.imshow('Noise', vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

if __name__ == '__main__':
    #print(__doc__)
    NoiseApp(0).run()