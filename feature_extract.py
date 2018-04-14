import cv2
import os
import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def normalize(vocab):
    std = StandardScaler().fit(vocab)
    vocab = std.transform(vocab)
    return vocab

#build histogram
def get_vocabb(n_images, descriptor_list, n_clusters, kmeans_ret, leng):
    kmeans_ret = list(kmeans_ret)
    hist = np.array([np.zeros(n_clusters) for i in range(n_images)])
    old_count = 0
    for i in range(n_images):
        l = list(leng)[i]
        for desc_idx in range(l):
            #print(kmeans_ret[old_count+desc_idx-1])
            #print(old_count+desc_idx)
            idx = kmeans_ret[old_count+desc_idx]
            ##add 1 to the hist of the ith image to the bin corresponding to class with index idx
            hist[i][idx] += 1
        old_count +=l
    print("hist generated")
    return hist

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

##for each clustersize build the separate histogram
k_best = [32, 64, 128, 256]
for k_best1 in k_best:
    kmeans = KMeans(n_clusters=k_best1)
    kmeans_ret1 = kmeans.fit_predict(descriptors)

    h5 = h5py.File('vocab'+str(k_best1)+'.h5', 'w')

    vocab1 = normalize(get_vocabb(len(lengthe), descriptors, k_best1, kmeans_ret1, lengthe))
    h5.create_dataset('vocab', data=vocab1)
    h5.close()
    print('vocab'+str(k_best1)+"finished"+"!!!!!!!")
