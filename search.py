import cv2
import os
import numpy as np
from scipy.spatial import distance
import h5py
from operator import itemgetter

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

def save_results(results, name):
    rs = {}
    labels = id_to_label(dirr)
    for key, val in results.items():
        rs[labels[key]] = [(vv[0], labels[vv[1]]) for vv in val]
    f = open(name, 'w')
    for key, values in rs.items():
        st = ''
        for t in values:
            st += str(t[0]) + " " + str(t[1]) + " "
        f.write(key + ": " + st + "\n")


def train(list_queries, vocab):
    #dictionary image_name: image_id
    dict1 = label_to_id(dirr)
    #construct the list of query images, get ids' for each image_name
    query_images = [dict1[q] for q in list_queries]
    results = {}
    #for each query image
    #find its' entry in the matrix of BoF of the corpus
    #vocab: rows - images, columns - features
    for q in query_images:
        #get features of query image
        features = vocab[q[0]:q[0] + 1, :]
        doc_res = [] # list of tuples for the query image [(dist, image1), ..., (dist, image1490)]
        #loop over all BoF of all images in the corpus
        for i, docs in enumerate(vocab):
            #compute the distances between query image and the image in corpus
            dist = distance(np.array(features), np.array(docs))
            doc_res.append((dist, i))
        results[q[0]] = doc_res
    res = {}
    #for each key - image sort it's value - the list of tuples by the first element - distance to the key
    for key, val in results.items():
        res[key] = sorted(val, key=itemgetter(0))
    return res



dirr = '~/the1_files/dataset/'

k_best = [32, 64, 128, 256]
with open('validation_queries.dat') as f:
    lines = f.readlines()
queries = [line.rstrip('\n') for line in open('validation_queries.dat')]

#for each number of clusters and BoF retrieved for them
for k in k_best:
    name = "vocab"+str(k)+'.h5'
    h5 = h5py.File(name, 'r')
    res = train(queries[:5], h5['vocab'][:])
    save_results(res, "results"+str(k)+".txt")
    print("results"+str(k)+"is written!!!!!!")
