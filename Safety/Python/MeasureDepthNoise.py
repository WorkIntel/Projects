
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

# local modules
from common import RectSelector
from opencv_viewer_depth import RealSense



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

        cv.namedWindow('Noise')
        self.rect_sel = RectSelector('Noise', self.on_rect)

    def define_roi(self, image, rect, data=None):
        '''define ROI target.'''
        x0, y0, x1, y1 = rect

    def statistics(self, frame):

        minv = frame.min()
        maxv = frame.max()
        avg  = frame.mean()
        stdv  = frame.std()
        locv = np.any(np.isnan(frame))
        print(f'Stat : {minv:.2f}:{avg:.2f}:{maxv:.2f} - std {stdv:.2f}3f. Has None {locv}')


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
        vis = self.frame.copy()
        #vis = cv.cvtColor(self.frame, cv.COLOR_GRAY2RGB)
        vis    = cv.convertScaleAbs(self.frame, alpha=0.03)
        self.rect_sel.draw(vis)

        if self.rect is not None:
            x0, y0, x1, y1 = self.rect
            vis = cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            vis = draw_str(vis,(x0,y0-10),str(err_std))

        return vis

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