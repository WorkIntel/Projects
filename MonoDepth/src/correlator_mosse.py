#!/usr/bin/env python

'''
MOSSE tracking sample including peak sub pixel

This sample implements correlation-based tracking approach, described in [1].

Usage:
  mosse.py [--pause] [<video source>]

  --pause  -  Start with playback paused at the first video frame.
              Useful for tracking target selection.

  Draw rectangles around objects with a mouse to track them.

Keys:
  SPACE    - pause video
  c        - clear targets

[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

from peak_fit_2d import peak_fit_2d
import time

import sys 
sys.path.append(r'C:\Users\udubin\Documents\Code\opencv-4x\samples\python')
from common import draw_str, RectSelector
import video
from mosse import MOSSE as MOSSE_ORIGIN

sys.path.append(r'..\Utils\src')
from opencv_realsense_camera import RealSense


def rnd_warp(a):
    h, w        = a.shape[:2]
    T           = np.zeros((2, 3))
    coef        = 0.2
    ang         = (np.random.rand()-0.5)*coef
    c, s        = np.cos(ang), np.sin(ang)
    T[:2, :2]   = [[c,-s], [s, c]]
    T[:2, :2]  += (np.random.rand(2, 2) - 0.5)*coef
    c           = (w/2, h/2)
    T[:,2]      = c - np.dot(T[:2, :2], c)
    return cv.warpAffine(a, T, (w, h), borderMode = cv.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi+1)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2  = rect
        w, h            = map(cv.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1          = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size       = w, h
        img             = cv.getRectSubPix(frame, (w, h), (x, y))

        self.win        = cv.createHanningWindow((w, h), cv.CV_32F)
        # g               = np.zeros((h, w), np.float32)
        # g[h//2, w//2]   = 1
        # g               = cv.GaussianBlur(g, (-1, -1), 2.0)
        # g              /= g.max()
        # self.G          = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)

        # self.H1         = np.zeros_like(self.G)
        # self.H2         = np.zeros_like(self.G)
        # for _i in xrange(16): #128):
        #     imgr        = rnd_warp(img)
        #     a           = self.preprocess(imgr)
        #     #a          = self.preprocess(img)
        #     A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
        #     self.H1    += A #cv.mulSpectrums(self.G, A, 0, conjB=True)
        #     self.H2    += cv.mulSpectrums(     A, A, 0, conjB=True)
        #     #self.H  = A

        self.init_kernel(img)
        self.update_kernel()
        self.update(frame)

    def init_kernel(self, img):
        "instead of init in the __init__ function"
        h, w            = img.shape
        self.H1         = np.zeros((h,w,2),dtype = np.float32 ) #np.zeros_like(self.G)
        self.H2         = np.zeros_like(self.H1)
        for _i in xrange(128): #128):
            imgr        = rnd_warp(img)
            a           = self.preprocess(imgr)
            A           = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            #A[0,0,:]    = eps # no DC
            #A[:,:,0],A[:,:,1] = A[:,:,0]*self.win,A[:,:,1]*self.win # no dc
            self.H1    += A #cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2    += cv.mulSpectrums(A, A, 0, conjB=True)

    def update_kernel(self):
        self.H          = divSpec(self.H1, self.H2)
        #self.H[...,1] *= -1            

    def update(self, frame, rate = 0.0): #125):
        (x, y), (w, h)      = self.pos, self.size
        self.last_img = img = cv.getRectSubPix(frame, (w, h), (x, y))
        img                 = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return

        self.pos            = x+dx, y+dy
        self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
        img                 = self.preprocess(img)
        A                   = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        #H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H1                  = A
        H2                  = cv.mulSpectrums(A, A, 0, conjB=True)
        self.H1             = self.H1 * (1.0-rate) + H1 * rate
        self.H2             = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()
        #self.H  = A

    @property
    def state_vis(self):
        f               = cv.idft(self.H, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT )
        h, w            = f.shape
        f               = np.roll(f, -h//2, 0)
        f               = np.roll(f, -w//2, 1)
        kernel          = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp            = self.last_resp
        resp            = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis             = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img         = np.log(np.float32(img)+1.0)
        img         = img-img.mean()  # important line
        #img         = (img-img.mean()) / (img.std()+eps)  # important line
        return img*self.win

    def correlate(self, img):
        A           = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        #A[:,:,0],A[:,:,1] = A[:,:,0]*self.win,A[:,:,1]*self.win # no dc
        C           = cv.mulSpectrums(A, self.H, 0, conjB=True)
        resp        = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        resp       = np.fft.fftshift(resp) # Shift the zero-frequency component to the center

        h, w        = resp.shape
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp   = resp.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr         = (mval-smean) / (sstd+eps)

        # sub pixel
        search_size = 3
        z           = resp[my-search_size:my+search_size, mx-search_size:mx+search_size]
        xp, yp      = peak_fit_2d(z)
        xp, yp      = xp - search_size, yp - search_size     
        #mx, my      = mx + xp, my + yp
        #print(f"{my:.2f},{mx:.2f}: {yp:.2f},{xp:.2f}")
        #print(f"{my:.2f},{mx:.2f}")
        #time.sleep(0.5)

        # # UD  subpixel
        # respc       = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_COMPLEX_OUTPUT)
        # respc       = np.fft.fftshift(respc)
        # resp        = cv.magnitude(respc[:,:,0],respc[:,:,1])   
        # cval        = respc[my,mx,:].squeeze()
        # angl        = np.arctan2(cval[1],cval[0])*180/np.pi
        # print(f"{my},{mx}: {cval[0]:.2f},{cval[1]:.2f} : {angl:.2f}")
        # side_resp = respc.copy()
        #cv.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)        

        return resp, (mx-w//2, my-h//2), psr



#%% ================
# Original App
class App:
    def __init__(self, video_src, paused = False):
        self.cap = cv.VideoCapture(video_src) #video.create_capture(video_src)
        if self.cap is None or not self.cap.isOpened():
            print('Warning: unable to open video source: ', video_src)
            return
        
        ret, self.frame = self.cap.read()
        cv.imshow('frame', self.frame)
        self.rect_sel = RectSelector('frame', self.onrect)
        self.trackers = []
        self.paused = paused

    def get_frame_gray(self, frame_type = 0):
        "extracts gray frame"
        if frame_type == 0:
            frame_gray = self.frame[:,:,0]
        elif frame_type == 1:
            frame_gray = self.frame[:,:,1]            
        else:
            frame_gray = cv.cvtColor(self.frame)
        return frame_gray 

    def onrect(self, rect):
        frame_gray  = self.get_frame_gray(0)
        tracker     = MOSSE(frame_gray, rect) #MOSSE_ORIGIN(frame_gray, rect)
        self.trackers.append(tracker)

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray  = self.get_frame_gray(1)
                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv.imshow('frame', vis)
            ch = cv.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []

# RealSense L or R App
class AppRS:
    def __init__(self, video_src = 'iid', paused = False):
        #self.cap = video.create_capture(video_src)
        self.cap        = RealSense(video_src)
        #self.cap        = RealSense('ggd')
        _, self.frame   = self.cap.read()
        frame_gray      = self.get_frame_gray(0)
        vis             = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR) 
        cv.imshow('frame left', vis)
        cv.imshow('frame right', vis)
        self.rect_sel   = RectSelector('frame left', self.onrect)
        self.trackers_l = []
        self.trackers_r = []
        self.paused     = paused
        self.update_rate= 0

    def get_frame_gray(self, frame_type = 0):
        "extracts gray frame"
        if frame_type == 0:
            frame_gray = self.frame[:,:,0]
        elif frame_type == 1:
            frame_gray = self.frame[:,:,1]            
        else:
            frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        return frame_gray       

    def onrect(self, rect):
        #frame_gray      = self.frame #cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        frame_gray      = self.get_frame_gray(0)
        tracker         = MOSSE(frame_gray, rect)
        #tracker         = MOSSE_ORIGIN(frame_gray, rect)
        self.trackers_l.append(tracker)
        #frame_gray      = self.get_frame_gray(1)
        tracker         = MOSSE(frame_gray, rect)
        self.trackers_r.append(tracker)        

    def run(self):
        while True:
            if not self.paused:
                ret, frame_c = self.cap.read()
                if not ret:
                    break
                self.frame  = frame_c #[:,:,0]

            frame_gray  = self.get_frame_gray(0)
            for tracker in self.trackers_l:
                tracker.update(frame_gray, rate = self.update_rate) 

            frame_gray  = self.get_frame_gray(1)
            for tracker in self.trackers_r:
                tracker.update(frame_gray, rate = self.update_rate)
                # search for the match pattern in x direction - disrepancy is big
                if not tracker.good and tracker.pos[0] > tracker.size[0]:
                    tracker.pos = (tracker.pos[0] - 1, tracker.pos[1])
            
            vis_l = cv.cvtColor(self.frame[:,:,0], cv.COLOR_GRAY2BGR) 
            for tracker in self.trackers_l:
                tracker.draw_state(vis_l)

            vis_r = cv.cvtColor(self.frame[:,:,1], cv.COLOR_GRAY2BGR) 
            for tracker in self.trackers_r:
                tracker.draw_state(vis_r)                

            if len(self.trackers_l) > 0:
                cv.imshow('tracker state', self.trackers_l[-1].state_vis)
            self.rect_sel.draw(vis_l)

            cv.imshow('frame left', vis_l)
            cv.imshow('frame right', vis_r)
            ch = cv.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers_l.pop()
                self.trackers_r.pop()
            if ch == ord('u'):  
                self.update_rate = 0.1 if self.update_rate < 0.001 else 0              

        cv.destroyAllWindows()
        
if __name__ == '__main__':
    print (__doc__)
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])
    opts = dict(opts)
    try:
        video_src = args[0]
    except:
        video_src = 'iid'

    #App(0, paused = '--pause' in opts).run()

    AppRS(video_src, paused = '--pause' in opts).run()