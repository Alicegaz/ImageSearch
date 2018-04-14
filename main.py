import cv2
import os
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.spatial import distance
import h5py
from operator import itemgetter
from sklearn.preprocessing import StandardScaler

def get_SIFT(gray_img):
  sift = cv2.xfeatures2d.SIFT_create()
  kp, desc = sift.detectAndCompute(gray_img, 0.3)
  return kp, desc

#histogram contains the frequency of occurence of each visual word - cluster
#vocabulary constists of a set of histograms of encompassing all descriptions for all images

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

def formatND(l):
    print(l.shape)
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
    print(vStack.shape)
    return vStack

index = []
dirr = '~/the1_files/dataset/'

def transform(img):
    img = cv2.imread(dirr+c)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k, des = sift.detectAndCompute(gray, None)
    descript_stack = formatND(descriptors)
    return des, descrit_stack


def save_results(results, name):
    rs = {}
    labels = id_to_label(dirr, damaged)
    for key, val in results.items():
        rs[labels[key]] = [(vv[0], labels[vv[1]]) for vv in val]
    f = open(name, 'w')
    for key, values in rs.items():
        st = ''
        for t in values:
            st += str(t[0]) + " " + str(t[1]) + " "
        f.write(key + ": " + st + "\n")


def normalize(vocab):
    std = StandardScaler().fit(vocab)
    vocab = std.transform(vocab)
    return vocab

def label_to_id(dirr, damaged):
    files = [f for f in os.listdir(dirr)[:] if f not in damaged]
    listt = {c: i for i, c in enumerate(files)}
    return listt
def id_to_label(dirr, damaged):
    files = [f for f in os.listdir(dirr)[:] if f not in damaged]
    listt = {i: c for i, c in enumerate(files)}
    return listt

def distance(a, b):
    a = a.T.reshape((a.T.shape[0], 1))
    b = b.reshape((b.shape[0], 1))
    v1 = np.array(a-b)
    return np.sqrt(np.sum((a-b)**2))

def search_method(query_images, vocab):
    results = {}
    #print(query_images)
    for q in query_images:
        #print(q)
        features = vocab[q[0]:q[0]+1, :]
        doc_res = []
        for i, docs in enumerate(vocab):
            dist = distance(np.array(features), np.array(docs))
            #dist docid
            doc_res.append((dist, i))
        results[q[0]] = doc_res
    res = {}
    #print('res', results)
    for key, val in results.items():
        #print(val)
        res[key] = sorted(val, key=itemgetter(0))
    return res

def train(list_queries, vocab):
    dict1 = label_to_id(dirr, damaged)
    ids_query = []
    query_images = []
    #for query in list_queries:
    #    id_query = [dict1[q] for q in list_queries]
    #    ids_query.append(id_query)
    #    query_images.append((dict1[query[0]], query))
    #ids_query = np.array(ids_query)
    query_images = [dict1[q] for q in list_queries]
    searchh = search_method(query_images, vocab)
    return searchh

damaged = []
kp = []
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
      kp.append(k)
      print('Processed image {} of {} damaged {}'.format(i, len(os.listdir(dirr)), len(damaged)))
    descriptors = descriptors[1:, :]
    h5 = h5py.File('descriptors_res.h5', 'w')
    h5.create_dataset('descriptors', data=descriptors)
    h5.create_dataset('lenghtes', data=np.array(desc_lengthes))
    h5.close()



#SIFT_descriptors(1490)
#h5 = h5py.File('~/descriptors_res.h5', 'r')
#descriptors = h5['descriptors'][:]
#h5.close()

h5 = h5py.File('~/descriptors_res.h5', 'r')
descriptors = h5['descriptors'][:]
lengthe = h5['lenghtes'][:]
h5.close()
descript_stack = descriptors

k_best = [32, 64, 128, 256]
for k_best1 in k_best:
    kmeans = KMeans(n_clusters=k_best1)
    kmeans_ret1 = kmeans.fit_predict(descript_stack)

    h5 = h5py.File('vocab'+str(k_best1)+'.h5', 'w')

    vocab1 = normalize(get_vocabb(len(lengthe), descriptors, k_best1, kmeans_ret1, lengthe))
    h5.create_dataset('vocab', data=vocab1)
    h5.close()
    print('vocab'+str(k_best1)+"finished"+"!!!!!!!")
with open('validation_queries.dat') as f:
    lines = f.readlines()
queries = [line.rstrip('\n') for line in open('validation_queries.dat')]
for k in k_best:
    name = "vocab"+str(k)+'.h5'
    h5 = h5py.File(name, 'r')
    vocab = h5['vocab'][:]
    res = train(queries, vocab)
    name = "results"+str(k)+".txt"
    save_results(res, name)
    print("results"+str(k)+"is written!!!!!!")
