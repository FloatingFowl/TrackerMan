import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Hog:

    def __init__(self, srcdir, stat):
        '''
        srcdir => directory with only image names, sorted by frame
        stat => display number of frames completed
        '''
        self.srcdir = srcdir
        self.imagelist = [f for f in os.listdir(srcdir) if not f.startswith('.')]
        self.imagelist.sort()
        self.status = stat

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.detector_all()

    def detector_all(self):
        self.dct = {}
        self.init_dict(self.dct)

        for i in range(len(self.imagelist)):
            if self.status and i % 25 == 0:
                print("Status", i, len(self.imagelist))
                
            det = self.detector_one(self.imagelist[i])

            if len(det.keys()) > 0:
                self.dct['fr'].extend([i+1 for _ in range(len(det['x']))])
                for j in det.keys():
                    self.dct[j].extend(det[j])

    def detector_all_parallel(self):
        #TODO
        pass

    def detector_one(self, imnm):
        im = cv2.imread(self.srcdir + imnm)

        #(rects, weights) = self.hog.detectMultiScale(im, winStride=(4, 4),
        #    padding=(8, 8), scale=1.05)
        (rects, weights) = self.hog.detectMultiScale(im, scale=1.01)

        ret = {}
        self.init_dict(ret)

        for i in range(len(weights)):
            ret['x'].append(rects[i][0])
            ret['y'].append(rects[i][1])
            ret['w'].append(rects[i][2])
            ret['h'].append(rects[i][3])
            ret['r'].append(weights[i][0])

        #display
        for (x, y, w, h) in rects:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        plt.imshow(im)
        plt.show()
        return ret

    def init_dict(self, dct):
        dct['x'] = []
        dct['y'] = []
        dct['w'] = []
        dct['h'] = []
        dct['r'] = []
        dct['fr'] = []
