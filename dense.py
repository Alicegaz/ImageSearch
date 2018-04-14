from dsift import *
import cv2
import numpy as np
import h5py
import os
from scipy import misc

dirr = 'dataset/'

def dense_sift(size):
    descriptors = []
    desc_lengthes = []
    for i, c in enumerate(os.listdir(dirr)[0:size]):
        img = cv2.imread(dirr + c)
        #if not isinstance(img, np.ndarray):
        #    img = misc.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        step = 16
        keypoints = []
        for l in range(int(step/2), gray.shape[0], step):
            for j in range(int(step/2), gray.shape[1], step):
                keypoints.append(cv2.KeyPoint(float(j), float(l), _size=float(step+4)))
        sift = cv2.xfeatures2d.SIFT_create()
        descriptors = []
        keypoints, desc = sift.compute(gray, keypoints)
        print(len(desc), len(desc[0]))
        desc_lengthes.append(desc.shape[0])
        #descriptors = np.concatenate((descriptors, desc), axis=0)
        descriptors.append(list(desc))
        print('Processed image {} of {}'.format(i, len(os.listdir(dirr))))
    #descriptors = descriptors[1:, :]
    descriptors = np.array(descriptors)
    h5 = h5py.File('ddescriptors_res.h5', 'w')
    h5.create_dataset('descriptors', data=descriptors)
    h5.create_dataset('lenghtes', data=np.array(desc_lengthes))
    h5.close()

dense_sift(1490)

