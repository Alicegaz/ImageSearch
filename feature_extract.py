import cv2
import os
import numpy as np
import h5py

damaged = []
dirr = '~/the1_files/dataset/'
def SIFT_descriptors(database_size):
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = np.zeros((1, 128))
    desc_lengthes = []
    for i, c in enumerate(os.listdir(dirr)[0:database_size]):
      img = cv2.imread(dirr+c)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      #cv2.imshow('w', gray)
      #cv2.waitKey()
      k, des = sift.detectAndCompute(gray, None)
      if(k==None or len(k)==0):
        #gray = cv2.cvtColor(cv2.imread(dirr+c), cv2.COLOR_BGR2GRAY)
        #cv2.imshow('w', gray)
        #cv2.waitKey()
        damaged.append(c)
        continue
      desc_lengthes.append(des.shape[0])
      descriptors = np.concatenate((descriptors, des), axis=0)
      print('Processed image {} of {} damaged {}'.format(i, len(os.listdir(dirr)), len(damaged)))
    descriptors = descriptors[1:, :]
    h5 = h5py.File('descriptors_res.h5', 'w')
    h5.create_dataset('descriptors', data=descriptors)
    h5.create_dataset('lenghtes', data=np.array(desc_lengthes))
    h5.close()

#running features extraction
#features - histogram
SIFT_descriptors(1490)
h5 = h5py.File('~/descriptors_res.h5', 'r')
descriptors = h5['descriptors'][:]
lengthe = h5['lenghtes'][:]
h5.close()
